import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.autograd import Variable
from ..box_utils import decode, nms, IoU
from data import v2 as cfg
import collections

class Detect(Function):
    """At test time, Detect is the final layer of SSD.    Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh, tub=0, tub_overlap=0.4, tub_generate_score=0.7):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']
        self.tub = tub
        if self.tub > 0:
            self.tubelets = [dict() for _ in range(self.num_classes)]
            self.ides = [None for _ in self.tubelets]
            self.history_max_ides = [-1 for _ in range(self.num_classes)]
            self.tub_overlap = tub_overlap
            self.loss_hold_len = 3
            self.tub_generate_score = tub_generate_score
            self.output = torch.zeros(1, self.num_classes, self.top_k, 6)
        else:
            self.output = torch.zeros(1, self.num_classes, self.top_k, 5)

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        self.output.zero_()
        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.t().contiguous().unsqueeze(0)
        else:
            conf_preds = conf_data.view(num, num_priors,
                                        self.num_classes).transpose(2, 1)
            self.output.expand_(num, self.num_classes, self.top_k, 5)
        # if init_tub:
        #     if self.tub > 0:
        #         self.tubelets = [dict() for _ in range(self.num_classes)]
        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            num_det = 0
            for cl in range(1, self.num_classes):
                if self.tub > 0:
                    ide_list = torch.cuda.FloatTensor(num_priors).fill_(-1)
                    if self.tubelets[cl]:
                        iou = IoU(decoded_boxes, self.tubelets[cl])
                        # iou = iou/iou.max()
                        iou_max, iou_max_idx = torch.max(iou, dim=1)
                        iou_mask = iou_max.gt(self.tub_overlap)
                        if iou_mask.sum() == 0:
                            continue
                        # score_mask = conf_scores[cl].gt(self.tubelet_generate_score)
                        ide_list[iou_mask] = self.ides[cl].index_select(0, iou_max_idx[iou_mask])
                        tub_score = torch.zeros(iou_mask.sum()).cuda()
                        for re in range(iou_mask.sum()):
                            t = self.tubelets[cl][ide_list[iou_mask][re]][0][:,0]
                            tub_score[re] = (torch.max(t) + torch.mean(t))/2
                        conf_scores[cl][iou_mask] =  conf_scores[cl][iou_mask]*iou_max[iou_mask] + tub_score*(1-iou_max[iou_mask])

                c_mask = conf_scores[cl].gt(self.conf_thresh)
                # a = c_mask.sum()
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    if self.tub > 0:
                        self.delete_tubelets(cl)
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                if self.tub > 0:
                    identity = ide_list[c_mask][ids[:count]]
                    new_mask = identity.eq(-1)
                    tub_score_mask = scores[ids[:count]].gt(self.tub_generate_score)
                    generate_mask = new_mask & tub_score_mask
                    if generate_mask.sum() > 0:
                        current =  0 if self.history_max_ides[cl]<0 else self.history_max_ides[cl]+1
                        new_id = torch.arange(current, current+generate_mask.sum())
                        self.history_max_ides[cl] = new_id[-1]
                        identity[generate_mask] = new_id.float()

                    self.output[i, cl, :count] = \
                        torch.cat((scores[ids[:count]].unsqueeze(1),
                                   boxes[ids[:count]],
                                   identity.unsqueeze(1)), 1)

                    for det in self.output[i, cl, :count]:
                        if det[-1] not in self.tubelets[cl]:
                            self.tubelets[cl][det[-1]] = [det[:-1].clone().unsqueeze(0), self.loss_hold_len+1]
                        else:
                            new_tube = torch.cat((det[:-1].clone().unsqueeze(0), self.tubelets[cl][det[-1]][0]), 0)
                            self.tubelets[cl][det[-1]] = [new_tube[:self.tub], self.loss_hold_len+1] if new_tube.size(0)>self.tub else [new_tube, self.loss_hold_len+1]
                    self.delete_tubelets(cl)
                else:
                    self.output[i, cl, :count] = \
                        torch.cat((scores[ids[:count]].unsqueeze(1),
                                   boxes[ids[:count]]), 1)
        flt = self.output.view(-1, self.output.size(-1))
        _, idx = flt[:, 0].sort(0)
        _, rank = idx.sort(0)
        flt[(rank >= self.top_k).unsqueeze(1).expand_as(flt)].fill_(0)
        return self.output

    def init_tubelets(self):
        if self.tub > 0:
            self.tubelets = [dict() for _ in range(self.num_classes)]

    def delete_tubelets(self, cl):
        delet_list = []
        for ide, tubelet in self.tubelets[cl].items():
            tubelet[-1] -= 1
            if not tubelet[-1]:
                delet_list.append(ide)
        # if not delet_list:
        #     return
        # else:
        for ide in delet_list:
            del self.tubelets[cl][ide]
        self.ides[cl] = torch.cuda.FloatTensor(list(self.tubelets[cl].keys()))

