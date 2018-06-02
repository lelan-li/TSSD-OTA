import torch
import torch.backends.cudnn as cudnn
from data import base_transform, VID_CLASSES, VID_CLASSES_name, MOT_CLASSES, UW_CLASSES
from model import build_ssd
from layers.modules import  AttentionLoss
import os
import numpy as np
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
dataset_name = 'UW'
backbone = 'VGG16'
ssd_dim=300
tub = 10
tub_thresh = 1
tub_generate_score = 0.1

if dataset_name == 'VID2017':
    model_dir='./weights/tssd300_VID2017_b8s8_RContiAttTBLstmAsso75_baseDrop2Clip5_FixVggExtraPreLocConf20000/ssd300_seqVID2017_5000.pth'
    # model_dir='./weights/ssd300_VIDDET_512/ssd300_VIDDET_5000.pth'
    video_name='/home/sean/data/ILSVRC/Data/VID/snippets/val/ILSVRC2015_val_00027000.mp4'
    labelmap = VID_CLASSES
    num_classes = len(VID_CLASSES) + 1
    prior = 'v2'
    confidence_threshold = 0.5
    nms_threshold = 0.5
    top_k = 200
elif dataset_name == 'MOT15':
    model_dir='./weights/tssd300_MOT15_SAL222/ssd300_seqMOT15_4000.pth'
    val_list = ['TUD-Campus.mp4', 'ETH-Sunnyday.mp4', 'ETH-Pedcross2.mp4', 'ADL-Rundle-8.mp4', 'Venice-2.mp4', 'KITTI-17.mp4']
    all_list = {0:'ADL-Rundle-1.mp4', 1:'ADL-Rundle-3.mp4', 2:'ADL-Rundle-6.mp4', 3:'ADL-Rundle-8.mp4', 4:'AVG-TownCentre.mp4',
                5:'ETH-Bahnhof.mp4', 6:'ETH-Crossing.mp4', 7:'ETH-Jelmoli.mp4', 8:'ETH-Linthescher.mp4', 9:'ETH-Pedcross2.mp4',
                10:'ETH-Sunnyday.mp4', 11:'PETS09-S2L1.mp4', 12:'PETS09-S2L2.mp4', 13:'TUD-Campus.mp4', 14:'TUD-Crossing.mp4',
                15:'TUD-Stadtmitte.mp4', 16:'Venice-1.mp4', 17:'Venice-2.mp4'}
    video_name = '/home/sean/data/MOT/snippets/'+all_list[7]
    labelmap = MOT_CLASSES
    num_classes = len(MOT_CLASSES) + 1
    prior = 'VOC_' + backbone + '_' + str(ssd_dim)
    confidence_threshold = 0.12
    nms_threshold = 0.3
    top_k = 400
elif dataset_name == 'UW':
    model_dir='./weights040/UW/tssd300_UW_SAL_816/ssd300_seqUW_5000.pth'
    # model_dir='./weights040/UW/ssd300_UW/ssd300_UW_30000.pth'
    labelmap = UW_CLASSES
    num_classes = len(UW_CLASSES) + 1
    prior = 'VOC_' + backbone + '_' + str(ssd_dim)
    confidence_threshold = 0.4
    nms_threshold = 0.3
    top_k = 400
    video_name='/home/sean/data/UWdevkit/Data/snippets/o20_6.wmv'#_GAN_RS.mp4'

else:
    raise ValueError("dataset [%s] not recognized." % dataset_name)

if model_dir.split('/')[-2].split('_')[0][0]=='t':
    tssd = 'tblstm'
    attention = True
else:
    tssd = 'ssd'
    attention = False

# save_dir = os.path.join('./demo/OTA', video_name.split('/')[-1].split('.')[0]+'_040')
save_dir = None
if save_dir and not os.path.exists(save_dir):
    os.mkdir(save_dir)

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def main():
    mean = (104, 117, 123)
    trained_model = model_dir

    print('loading model!')
    net = build_ssd('test', ssd_dim, num_classes, tssd=tssd,
                    top_k=top_k,
                    thresh=confidence_threshold,
                    nms_thresh=nms_threshold,
                    attention=attention,
                    prior=prior,
                    tub = tub,
                    tub_thresh = tub_thresh,
                    tub_generate_score=tub_generate_score)
    net.load_state_dict(torch.load(trained_model))
    net.eval()

    print('Finished loading model!', model_dir)

    net = net.cuda()
    cudnn.benchmark = True

    frame_num = 9900
    cap = cv2.VideoCapture(video_name)
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(w, h)
    size = (640, 480)
    if save_dir:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        record = cv2.VideoWriter(os.path.join(save_dir,video_name.split('/')[-1].split('.')[0]+'_OTA.avi'), fourcc, cap.get(cv2.CAP_PROP_FPS), size)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    att_criterion = AttentionLoss((h, w))
    state = [None] * 6 if tssd in ['lstm', 'tblstm'] else None
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame_draw = frame.copy()
        frame_num += 1
        im_trans = base_transform(frame, ssd_dim, mean)
        with torch.no_grad():
            x = torch.from_numpy(im_trans).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            if tssd == 'ssd':
                detections, att_map = net(x)
            else:
                detections, state, att_map = net(x, state)
        out = list()
        for j in range(1, detections.size(1)):
            if detections[0, j, :, :].sum() == 0:
                continue
            for k in range(detections.size(2)):
                dets = detections[0, j, k, :]
                if dets.sum() == 0:
                    continue
                boxes = dets[1:-1] if dets.size(0) == 6 else dets[1:]
                identity = dets[-1] if dets.size(0) == 6 else -1
                x_min = int(boxes[0] * w)
                x_max = int(boxes[2] * w)
                y_min = int(boxes[1] * h)
                y_max = int(boxes[3] * h)

                score = dets[0]
                if score > confidence_threshold:
                    out.append([x_min, y_min, x_max, y_max, j - 1, score.cpu().numpy(), identity])

        if attention:
            _, up_attmap = att_criterion(att_map)  # scale, batch, tensor(1,h,w)
            att_target = up_attmap[0][0].cpu().data.numpy().transpose(1, 2, 0)
        for object in out:
            x_min, y_min, x_max, y_max, cls, score, identity = object
            if dataset_name in ['MOT15']:
                put_str = str(int(identity))
                if identity in [34]:
                    color = (0, 0, 255)
                elif identity in [35]:
                    color = (0, 200, 0)
                elif identity in [58]:
                    color = (255, 0, 255)
                # elif identity in [3]:
                #     color = (255, 0, 255)
                # elif identity in [4]:
                #     color = (0, 128, 255)
                # elif identity in [5]:
                #     color = (255, 128, 128)
                else:
                    color = (255, 0, 0)
            elif dataset_name in ['VID2017']:
                put_str = str(int(identity))+':'+VID_CLASSES_name[cls] +':'+ str(np.around(score, decimals=2))
            elif dataset_name in ['UW']:
                put_str = str(int(identity))
                if cls == 0:
                    color = (min(int(identity)+1, 255), 0, 255)
                elif cls == 1:
                    color = (255, min(int(identity)+1, 255), 0)
                elif cls == 2:
                    color = (min(int(identity)+1, 255), 128, 0)

            cv2.rectangle(frame_draw, (x_min, y_min), (x_max, y_max), color, thickness=2)
            cv2.fillConvexPoly(frame_draw, np.array(
                [[x_min - 1, y_min], [x_min - 1, y_min - 50], [x_max + 1, y_min - 50], [x_max + 1, y_min]], np.int32),
                               color)

            cv2.putText(frame_draw, put_str,
                        (x_min + 10, y_min - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, color=(255, 255, 255), thickness=1)
            print(str(frame_num) + ':' + str(np.around(score, decimals=2)) + ','+VID_CLASSES_name[cls])
        if not out:
            print(str(frame_num))
        frame_show = cv2.resize(frame_draw, size)
        cv2.imshow('frame', frame_show)
        # cv2.imshow('att', cv2.resize(att_target, size))

        if save_dir:
            record.write(frame_show)
        ch = cv2.waitKey(1)
        if ch == 32:
        # if frame_num in [44]:
            while 1:
                in_ch = cv2.waitKey(10)
                if in_ch == 115: # 's'
                    if save_dir:
                        print('save: ', frame_num)
                        torch.save(out, os.path.join(save_dir, tssd+'_%s.pkl' % str(frame_num)))
                        cv2.imwrite(os.path.join(save_dir, '%s.jpg' % str(frame_num)), frame)
                elif in_ch == 32:
                    break

    cap.release()
    if save_dir:
        record.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

