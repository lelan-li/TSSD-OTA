import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import base_transform, VID_CLASSES, VID_CLASSES_name, MOT_CLASSES
from ssd import build_ssd
from layers.modules import  AttentionLoss
import os
import numpy as np
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
model_dir='./weights/tssd300_VID2017_b8s8_RContiAttTBLstmAsso75_baseDrop2Clip5_FixVggExtraPreLocConf20000'
# model_dir='./weights/tssd300_VID2017_b8s8_RSkipAttTBLstm_baseDrop2Clip5Refine_FixVggExtraPreLocConf'
# model_dir='./weights/ssd300_VIDDET_512'
model_name= 'ssd300'
literation='5000'
confidence_threshold=0.6
nms_threshold =0.45
top_k=200
ssd_dim=300
dataset_name = 'VID2017'
video_name='/home/sean/data/ILSVRC/Data/VID/snippets/val/ILSVRC2015_val_00027000.mp4'
save_dir = os.path.join('./demo/comp', video_name.split('/')[-1].split('.')[0])
print(save_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if model_dir.split('/')[2].split('_')[0][0]=='t':
    tssd = 'tblstm'
    attention = True
else:
    tssd = 'ssd'
    attention = False

refine = False
tub = 10
tub_overlap = 0.4
tub_generate_score = 0.7

torch.set_default_tensor_type('torch.cuda.FloatTensor')
if dataset_name in ['VID2017', 'seqVID2017']:
    labelmap = VID_CLASSES
    num_classes = len(VID_CLASSES) + 1
elif dataset_name in ['MOT17Det', 'seqMOT17Det' ]:
    labelmap = MOT_CLASSES
    num_classes = len(MOT_CLASSES) + 1
    # print(num_classes)
else:
    raise ValueError("dataset [%s] not recognized." % dataset_name)



def main():
    mean = (104, 117, 123)

    if model_dir.split('/')[-1] in ['ssd300_VIDDET',  'ssd300_VIDDET_512',
                     'attssd300_VIDDET_512']:
        trained_model = os.path.join(model_dir, 'ssd300_VIDDET_' + literation + '.pth')
    else:
        trained_model = os.path.join(model_dir,
                                     model_name + '_' + 'seq' + dataset_name + '_' + literation + '.pth') \
            if tssd in ['lstm', 'tblstm', 'outlstm'] else os.path.join(model_dir,
                                                                       model_name + '_' + dataset_name + '_' + literation + '.pth')

    print('loading model!')
    net = build_ssd('test', ssd_dim, num_classes, tssd=tssd,
                    top_k=top_k,
                    thresh=confidence_threshold,
                    nms_thresh=nms_threshold,
                    attention=attention,
                    refine=refine,
                    tub = tub,
                    tub_overlap = tub_overlap,
                    tub_generate_score=tub_generate_score)
    net.load_state_dict(torch.load(trained_model))
    net.eval()

    print('Finished loading model!', model_dir, literation)

    net = net.cuda()
    cudnn.benchmark = True

    frame_num = 0
    cap = cv2.VideoCapture(video_name)
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(w, h)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    att_criterion = AttentionLoss((h, w))
    state = [None] * 6 if tssd in ['lstm', 'tblstm', 'outlstm'] else None
    # pre_att_roi = list()
    # id_pre_cls = [0] * len(labelmap)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame_draw = frame.copy()
        frame_num += 1
        # objects, state, att_map = test_net(net, frame, w, h, state=state, thresh=confidence_threshold)
        im_trans = base_transform(frame, ssd_dim, mean)
        x = Variable(torch.from_numpy(im_trans).unsqueeze(0).permute(0, 3, 1, 2), volatile=True)
        x = x.cuda()
        if tssd == 'ssd':
            detections, att_map = net(x)
            detections = detections.data
        else:
            detections, state, att_map = net(x, state)
            detections = detections.data
            # print(np.around(t_diff, decimals=4))
        out = list()
        for j in range(1, detections.size(1)):
            for k in range(detections.size(2)):
                dets = detections[0, j, k, :]
                # mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                # dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.dim() == 0:
                    continue

                boxes = dets[1:-1] if dets.size(0) == 6 else dets[1:]
                identity = dets[-1] if dets.size(0) == 6 else -1
                x_min = int(boxes[0] * w)
                x_max = int(boxes[2] * w)
                y_min = int(boxes[1] * h)
                y_max = int(boxes[3] * h)

                score = dets[0]
                if score > confidence_threshold:
                    out.append([x_min, y_min, x_max, y_max, j - 1, score, identity])

        if attention:
            _, up_attmap = att_criterion(att_map)  # scale, batch, tensor(1,h,w)
            att_target = up_attmap[0][0].cpu().data.numpy().transpose(1, 2, 0)
        for object in out:
            color = (0, 0, 255)
            x_min, y_min, x_max, y_max, cls, score, identity = object
            cv2.rectangle(frame_draw, (x_min, y_min), (x_max, y_max), color, thickness=2)
            cv2.fillConvexPoly(frame_draw, np.array(
                [[x_min - 1, y_min], [x_min - 1, y_min - 50], [x_max + 1, y_min - 50], [x_max + 1, y_min]], np.int32),
                               color)
            cv2.putText(frame_draw, str(int(identity))+':'+VID_CLASSES_name[cls] +':'+ str(np.around(score, decimals=2)),
                        (x_min + 10, y_min - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color=(255, 255, 255), thickness=1)
            print(str(frame_num) + ':' + str(np.around(score, decimals=2)) + ','+VID_CLASSES_name[cls])
        if not out:
            print(str(frame_num))
        cv2.imshow('frame', frame_draw)
        ch = cv2.waitKey(1)
        if ch == 32:
        # if frame_num in [72, 113]:
            while 1:
                in_ch = cv2.waitKey(10)
                if in_ch == 115: # 's'
                    if save_dir:
                        print('save: ', frame_num)
                        torch.save(out, os.path.join(save_dir, tssd+'_%s.pkl' % str(frame_num)))
                        # if tssd == 'ssd':
                        #     save_tuple =(out, up_attmap) if attention else (out,)
                        #     torch.save(save_tuple, os.path.join(save_dir, 'ssd_%s.pkl' % str(frame_num)))
                        # else:
                        cv2.imwrite(os.path.join(save_dir, '%s.jpg' % str(frame_num)), frame)
                        #     save_tuple =(out,state, up_attmap) if attention else (out, state)
                        #     torch.save(save_tuple, os.path.join(save_dir, '%s.pkl' % str(frame_num)))
                        # cv2.imwrite('./11.jpg', frame)
                elif in_ch == 32:
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

