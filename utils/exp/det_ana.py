import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import os
import cv2

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

VID_CLASSES_name =(  # always index 0
    'airplane', #1 airplane
    'antelope', #2 antelope
    'bear', #3 bear
    'bicycle', #4 bicycle
    'bird', #5 bird
    'bus', #6 bus
    'car', #7 car
    'cattle', #8 cattle
    'dog', #9 dog
    'domestic_cat', #10 domestic_cat
    'elephant', #11 elephant
    'fox', #12 fox
    'giant_panda', #13 giant_panda
    'hamster', #14 hamster
    'horse', #15 horse
    'lion', #16 lion
    'lizard', #17 lizard
    'monkey', #18 monkey
    'motorcycle', #19 motorcycle
    'rabbit', #20 rabbit
    'red_panda', #21 red_panda
    'sheep', #22 sheep
    'snake', #23 snake
    'squirrel', #24 squirrel
    'tiger', #25 tiger
    'train', #26 train
    'turtle', #27 turtle
    'watercraft', #28 watercraft
    'whale', #29 whale
    'zebra', #30 zebra
)
color = {'car':'b', 'watercraft':'m', 'dog':'r', 'whale':'g', 'airplane':'g'}

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
dir = '../../demo/comp'

font_text = {'family' : 'Arial',
        'color'  : 'w',
        'weight' : 'normal',
        'size'   : 8,
        }
font_label = {'family' : 'Arial',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 10,
        }
font_single = {'family' : 'Arial',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 20,
        }
targets = ['ILSVRC2015_val_00007010', 'ILSVRC2015_val_00036000', 'ILSVRC2015_val_00049000', 'ILSVRC2015_val_00020001']

frames = [[10,45,83,90,112], [17,23,42,60,76],[3,10,12,19,71], [3, 29,46, 55, 113]]

color_id = {-1.0:'g', 0.0:'r', 1.0:'m', 2.0:'w', 3.0:'c', 4.0:'b', 5.0:'orange', 6.0:'y'}

gs = gridspec.GridSpec(len(targets)*2, len(frames[0]))

for i, (target,frame) in enumerate(zip(targets,frames)):
    if i==0:
        frame = frame[::-1]
    for j, f in enumerate(frame):
        img = cv2.cvtColor(cv2.imread(os.path.join(dir, target, str(f) + '.jpg')), cv2.COLOR_BGR2RGB)
        objects = torch.load(os.path.join(dir, target, 'tblstm_'+str(f)+'.pkl'))
        ssd_objects = torch.load(os.path.join(dir, target, 'ssd_'+str(f)+'.pkl'))
        if i==0:
            ax = plt.subplot(gs[2*i, len(frame)-j-1])
        else:
            ax = plt.subplot(gs[2*i, j])
        plt.imshow(img)
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_xticks([])
        ax.set_yticks([])
        if i==0:
            if j==len(frames[0])-1:
                plt.ylabel('SSD', fontdict=font_label)
        else:
            if j==0:
                plt.ylabel('SSD', fontdict=font_label)
        for object in ssd_objects:
            x_min, y_min, x_max, y_max, cls, score, id = object
            score = np.around(score,decimals=2)
            color = color_id[id] #color[VID_CLASSES_name[cls]]
            rect = patches.Rectangle((x_min, y_min), x_max - x_min,
                                 y_max - y_min, linewidth=1.0, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            if i == 0:
                t = plt.text(x_min, y_min, str(int(id))+':'+VID_CLASSES_name[cls]+':'+str(score), color=color, fontdict=font_text)
                t.set_bbox(dict(facecolor=color, alpha=0.1, edgecolor=color))
            else:
                t = plt.text(x_min, y_min, str(int(id))+':'+VID_CLASSES_name[cls]+':'+str(score), color='w', fontdict=font_text)
                t.set_bbox(dict(facecolor=color, alpha=0.5, edgecolor=color))

        if i==0:
            ax = plt.subplot(gs[2*i+1, len(frame)-j-1])
        else:
            ax = plt.subplot(gs[2*i+1, j])
        plt.imshow(img)
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_xticks([])
        ax.set_yticks([])
        if i==0:
            if j==len(frames[0])-1:
                plt.ylabel('TSSD-OTA', fontdict=font_label)
        else:
            if j==0:
                plt.ylabel('TSSD-OTA', fontdict=font_label)
        for object in objects:
            x_min, y_min, x_max, y_max, cls, score, id = object
            if i==1:
                id -= 1
            score = np.around(score,decimals=2)
            color = color_id[id] #color[VID_CLASSES_name[cls]]
            rect = patches.Rectangle((x_min, y_min), x_max - x_min,
                                 y_max - y_min, linewidth=1.0, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            if i == 0:
                t = plt.text(x_min, y_min, str(int(id))+':'+VID_CLASSES_name[cls]+':'+str(score), color=color, fontdict=font_text)
                t.set_bbox(dict(facecolor=color, alpha=0.1, edgecolor=color))
            else:
                t = plt.text(x_min, y_min, str(int(id))+':'+VID_CLASSES_name[cls]+':'+str(score), color='w', fontdict=font_text)
                t.set_bbox(dict(facecolor=color, alpha=0.5, edgecolor=color))
plt.show()

# img = cv2.cvtColor(cv2.imread(os.path.join(dir, targets[0],'10.jpg')), cv2.COLOR_BGR2RGB)
# objects = torch.load(os.path.join(dir, targets[0], 'tblstm_10.pkl'))
#
# ax = plt.subplot(1,1,1)
# plt.imshow(img[:,400:-200])
# plt.axis('off')

# for object in objects:
#     x_min, y_min, x_max, y_max, cls, score, id = object
#     x_min -= 400
#     x_max-=400
#     score = np.around(score, decimals=2)
#     rect = patches.Rectangle((x_min, y_min), x_max - x_min,
#                              y_max - y_min, linewidth=2.0, edgecolor=color[VID_CLASSES_name[cls]], facecolor='none')
#     ax.add_patch(rect)
#     t = plt.text(x_min, y_min, VID_CLASSES_name[cls] + ':' + str(score), color=color[VID_CLASSES_name[cls]], # str(int(id))+':'+
#                  fontdict=font_single)
#     t.set_bbox(dict(facecolor='w', alpha=0.1, edgecolor=color[VID_CLASSES_name[cls]]))
# ax.spines['right'].set_color('none')
# ax.spines['left'].set_color('none')
# ax.spines['bottom'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.set_xticks([])
# ax.set_yticks([])
# plt.show()

