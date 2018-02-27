from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from data import VOCroot, VIDroot
# from data import VOC_CLASSES as labelmap
import torch.utils.data as data

from data import base_transform, VID_CLASSES, VID_CLASSES_name
from ssd import build_ssd
from layers.modules import  AttentionLoss


import sys
import os
import time
import argparse
import numpy as np
import cv2
import matplotlib

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--model_name', default='ssd300',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--confidence_threshold', default=0.3, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--nms_threshold', default=0.45, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=10, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--dataset_name', default='seqVID2017', help='Which dataset')
parser.add_argument('--ssd_dim', default=300, type=int, help='ssd_dim 300 or 512')
parser.add_argument('--literation', default='2900000', type=str,help='File path to save results')
parser.add_argument('--model_dir', default='./weights/ssd300_VID2017', type=str,help='Path to save model')
parser.add_argument('--video_name', default='/home/sean/data/ILSVRC/Data/VID/snippets/val/ILSVRC2017_val_00131000.mp4', type=str,help='Path to video')
parser.add_argument('--tssd',  default='ssd', type=str, help='ssd or tssd')
parser.add_argument('--gpu_id',  default='1', type=str, help='gpu_id')
parser.add_argument('--attention', default=False, type=str2bool, help='attention')
parser.add_argument('--save_dir',  default=None, type=str, help='save dir')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.calls += 1
        if self.calls > 10:
            self.diff = time.time() - self.start_time
            self.total_time += self.diff
            self.average_time = self.total_time / (self.calls-10)
            if average:
                return self.average_time
            else:
                return self.diff
        else:
            return 0.

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if args.dataset_name == 'VID2017' or 'seqVID2017':
    labelmap = VID_CLASSES
    num_classes = len(VID_CLASSES) + 1
else:
    raise ValueError("dataset [%s] not recognized." % args.dataset_name)


def test_net(net, im, w, h, state=None, thresh=0.5, tim=None):
    im_trans = base_transform(im, ssd_dim, mean)
    x = Variable(torch.from_numpy(im_trans).unsqueeze(0).permute(0, 3, 1, 2), volatile=True)
    if args.cuda:
        x = x.cuda()
    if args.tssd == 'ssd':
        detections, att_map = net(x)
        detections = detections.data
    else:
        tim.tic()
        detections, state, att_map = net(x, state)
        detections = detections.data
        t_diff = tim.toc(average=True)
        # print(np.around(t_diff, decimals=4))
    out = list()
    for j in range(1, detections.size(1)):
        for k in range(detections.size(2)):
            dets = detections[0, j, k, :]
            # mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            # dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            boxes = dets[1:]
            x_min = int(boxes[0] * w)
            x_max = int(boxes[2] * w)
            y_min = int(boxes[1] * h)
            y_max = int(boxes[3] * h)

            score = dets[0]
            if score > thresh:
                out.append([x_min, y_min, x_max, y_max, j-1, score])

    return tuple(out), state, att_map

def att_match(att_roi_tuple, pre_att_roi_tuple, pooling_size=30):
    match_list = [None] * len(att_roi_tuple)
    if not pre_att_roi_tuple:
        return match_list
    else:
        xycls_dis = np.zeros(len(att_roi_tuple), len(pre_att_roi_tuple))
        for num, obj in enumerate(att_roi_tuple):
            obj[0] = [F.upsample(roi, (pooling_size,pooling_size), mode='bilinear') for roi in obj[0]]
            obj_x_min, obj_y_min, obj_x_max, obj_y_max, obj_cls = obj[1:]
            for pre_num, pre_obj in enumerate(pre_att_roi_tuple):
                if pre_num == 0:
                    pre_obj[0] = [F.upsample(preroi, (pooling_size,pooling_size)) for preroi in pre_att_roi]
                preobj_x_min, preobj_y_min, preobj_x_max, preobj_y_max, preobj_cls = pre_obj[1:]
                xycls_dis[num, pre_num] = (obj_x_min - preobj_x_min) + \
                                          (obj_y_min - preobj_y_min) + \
                                          (obj_x_max - preobj_x_max) + \
                                          (obj_y_max - preobj_y_max) + \
                                          (1,0)[obj_cls==preobj_cls]

        return match_list

if __name__ == '__main__':

    mean = (104, 117, 123)
    ssd_dim = args.ssd_dim

    if args.model_dir in ['../weights/ssd300_VIDDET', '../weights/ssd300_VIDDET_186', '../weights/ssd300_VIDDET_512', '../weights/attssd300_VIDDET_512']:
        trained_model = os.path.join(args.model_dir, 'ssd300_VIDDET_' + args.literation + '.pth')
    else:
        trained_model = os.path.join(args.model_dir,
                                     args.model_name + '_' + 'seq' + args.dataset_name + '_' + args.literation + '.pth') \
            if args.tssd in ['lstm', 'tblstm', 'outlstm'] else os.path.join(args.model_dir,
                                                       args.model_name + '_' + args.dataset_name + '_' + args.literation + '.pth')

    print('loading model!')
    net = build_ssd('test', ssd_dim, num_classes, tssd=args.tssd,
                    top_k = args.top_k,
                    thresh = args.confidence_threshold,
                    nms_thresh = args.nms_threshold,
                    attention=args.attention)
    net.load_state_dict(torch.load(trained_model))
    net.eval()

    print('Finished loading model!', args.model_dir, args.literation)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    tim = Timer()

    frame_num = 0
    cap = cv2.VideoCapture(args.video_name)
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(w,h)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    att_criterion = AttentionLoss((h,w))
    state = [None]*6 if args.tssd in ['lstm', 'tblstm', 'outlstm'] else None
    pre_att_roi = list()
    id_pre_cls = [0] * len(VID_CLASSES_name)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame_draw = frame.copy()
        frame_num += 1
        # if args.tssd == 'ssd':
        #     im_detect = test_net(net, frame, w, h, thresh=args.confidence_threshold, tim=tim)
        # else:
        objects, state, att_map = test_net(net, frame, w, h, state=state, thresh=args.confidence_threshold, tim=tim)
        if args.attention:
            _, up_attmap = att_criterion(att_map) # scale, batch, tensor(1,h,w)
            att_target = up_attmap[0][0].cpu().data.numpy().transpose(1, 2, 0)
        # print(up_attmap[0][0])
        att_roi = list()
        # if objects:
        for object in objects:
                # x_min, y_min, x_max, y_max, cls, score = object
                # if frame_num in [45, 55, 65]:
                #     print(x_min,y_min,x_max,y_max)
            # roi = frame[y_min:y_max,x_min:x_max]
            # att_roi_obj=[None]*len(up_attmap)
            # for scale in up_attmap:
                # att_roi_obj[scale] = up_attmap[scale][0][:,y_min:y_max,x_min:x_max]
            # att_roi.append([att_roi_obj,x_min/w,y_min/h,x_max/w,y_max/h,cls]) # [object[[roi], x, y, x, y, cls]]

        # match_list = att_match(att_roi, pre_att_roi)
        # pre_att_roi = att_roi

        # Draw
        # for object in objects:
                # if frame_num==55:
                #     color = (180,150,0)
                # else:
            color = (0,0,255)
            x_min, y_min, x_max, y_max, cls, score = object
            cv2.rectangle(frame_draw, (x_min, y_min), (x_max, y_max), color, thickness=2)
            cv2.fillConvexPoly(frame_draw, np.array(
            [[x_min-1, y_min], [x_min-1, y_min - 50], [x_max+1 , y_min - 50], [x_max+1, y_min]], np.int32),
                               color)
            cv2.putText(frame_draw, VID_CLASSES_name[cls] + str(np.around(score, decimals=2)),
                    (x_min + 10, y_min - 10), cv2.FONT_HERSHEY_DUPLEX, 1.4, color=(255, 255, 255), thickness=2)
            print(str(frame_num)+':'+str(np.around(score, decimals=2))+',')
        # cv2.imshow('roi', att_roi.cpu().data.numpy().transpose(1, 2, 0))
        # cv2.imshow('mask', att_target)
        # else:
        #     print(frame_num)
        cv2.imshow('frame', frame_draw)
        ch = cv2.waitKey(1)
        # if ch == 32:
        if frame_num in [1, 20]:
            while 1:
                in_ch = cv2.waitKey(10)
                if in_ch == 115: # 's'
                    if args.save_dir:
                        print('save: ', frame_num)
                        if args.tssd == 'ssd':
                            torch.save((objects, up_attmap), os.path.join(args.save_dir, 'ssd_%s.pkl' % str(frame_num)))
                        else:
                            cv2.imwrite(os.path.join(args.save_dir, '%s.jpg' % str(frame_num)), frame)
                            torch.save((objects, up_attmap), os.path.join(args.save_dir, '%s.pkl' % str(frame_num)))
                        # cv2.imwrite('./11.jpg', frame)
                elif in_ch == 32:
                    break


    cap.release()
    cv2.destroyAllWindows()
