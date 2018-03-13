import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.autograd import Variable
from ..box_utils import decode, nms, IoU
from data import v2 as cfg
import collections

class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh, tub=0):
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
                        # for ide, tube in self.tubelets[cl].items():
                        iou = IoU(decoded_boxes, self.tubelets[cl])
                        iou = iou/iou.max()
                        iou_max, iou_max_idx = torch.max(iou, dim=1)
                        iou_mask = iou_max.gt(0.65)
                        ide_list[iou_mask] = iou_max_idx[iou_mask].float()
                        tub_score = torch.zeros(iou_mask.sum()).cuda()
                        for re in range(iou_mask.sum()):
                            t = self.tubelets[cl][ide_list[iou_mask][re]][:,0]
                            tub_score[re] = (torch.max(t) + torch.mean(t))/2
                        conf_scores[cl][iou_mask] =  (conf_scores[cl][iou_mask] + tub_score)/2

                c_mask = conf_scores[cl].gt(self.conf_thresh)
                # a = c_mask.sum()
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                if self.tub > 0:
                    identity = ide_list[c_mask][ids[:count]]
                    new_mask = identity.eq(-1)
                    if new_mask.sum() > 0:
                        object_num =  len(self.tubelets[cl])
                        new_id = torch.arange(object_num, object_num+new_mask.sum())
                        identity[new_mask] = new_id.float()

                    self.output[i, cl, :count] = \
                        torch.cat((scores.unsqueeze(1)[ids[:count]].unsqueeze(1),
                                   boxes[ids[:count]],
                                   identity.unsqueeze(1)), 1)

                    for det in self.output[i, cl, :count]:
                        if det[-1] not in self.tubelets[cl]:
                            self.tubelets[cl][det[-1]] = det[:-1].clone().unsqueeze(0)
                        else:
                            self.tubelets[cl][det[-1]] = torch.cat((det[:-1].clone().unsqueeze(0), self.tubelets[cl][det[-1]]), 0)
                            if self.tubelets[cl][det[-1]].size(0) > self.tub:
                                self.tubelets[cl][det[-1]] = self.tubelets[cl][det[-1]][:self.tub]
                else:
                    self.output[i, cl, :count] = \
                        torch.cat((scores[ids[:count]].unsqueeze(1),
                                   boxes[ids[:count]]), 1)
        flt = self.output.view(-1, self.output.size(-1))
        _, idx = flt[:, 0].sort(0)
        _, rank = idx.sort(0)
        flt[(rank >= self.top_k).unsqueeze(1).expand_as(flt)].fill_(0)
        return self.output
