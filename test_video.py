from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOCroot, VIDroot
# from data import VOC_CLASSES as labelmap
import torch.utils.data as data

from data import base_transform, VID_CLASSES, VID_CLASSES_name
from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import cv2

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
        detections = net(x).data
    else:
        tim.tic()
        detections, state = net(x, state)
        detections = detections.data
        t_diff = tim.toc(average=True)
        print(np.around(t_diff, decimals=4))
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

            scores = dets[0]
            if scores > thresh:
                cv2.rectangle(im, (x_min, y_min), (x_max, y_max), (0,0,255), thickness=2)
                # cv2.rectangle(im, (x_min, y_min-30), (x_max, y_min), (0,0,255), thickness=2)
                cv2.fillConvexPoly(im, np.array([[x_min, y_min], [x_min, y_min+30], [x_max-30, y_min+30],[x_max-30, y_min]], np.int32), (0,0,255))
                cv2.putText(im, VID_CLASSES_name[j-1]+':'+str(np.around(scores,decimals=2)),
                            (x_min+10, y_min+18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(255,255,255), thickness=1)
    return im, state


if __name__ == '__main__':

    mean = (104, 117, 123)
    ssd_dim = args.ssd_dim

    if args.model_dir == '../weights/ssd300_VIDDET':
        trained_model = os.path.join(args.model_dir, args.model_dir.split('/')[-1] + '_' + args.literation + '.pth')
    else:
        trained_model = os.path.join(args.model_dir,
                                     args.model_name + '_' + 'seq' + args.dataset_name + '_' + args.literation + '.pth') \
            if args.tssd in ['lstm'] else os.path.join(args.model_dir,
                                                       args.model_name + '_' + args.dataset_name + '_' + args.literation + '.pth')

    print('loading model!')
    net = build_ssd('test', ssd_dim, num_classes, tssd=args.tssd,
                    top_k = args.top_k,
                    thresh = args.confidence_threshold,
                    nms_thresh = args.nms_threshold)
    net.load_state_dict(torch.load(trained_model))
    net.eval()
    print('Finished loading model!')
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    tim = Timer()

    cap = cv2.VideoCapture(args.video_name)
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    state = [None]*6 if args.tssd in ['lstm'] else None
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        # if args.tssd == 'ssd':
        #     im_detect = test_net(net, frame, w, h, thresh=args.confidence_threshold, tim=tim)
        # else:
        im_detect, state = test_net(net, frame, w, h, state=state, thresh=args.confidence_threshold, tim=tim)
        cv2.imshow('frame', im_detect)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
