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
color = {'car':'r', 'horse':'c', 'lizard':'b', 'whale':'g', 'airplane':'m'}


os.environ["CUDA_VISIBLE_DEVICES"] = '3'
dir = '../demo/res'

font_text = {'family' : 'Arial',
        'color'  : 'w',
        'weight' : 'normal',
        'size'   : 8,
        }
font_label = {'family' : 'Arial',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 18,
        }
targets = ['car11000', 'horse47000', 'whale36000', 'lizard13001','airplane7010']

frames = [[27,37,54,75,88], [15,61,104,256,404],[18,34,60,76,89], [75, 168,304,361,385,413]
         , [8,12,39,64,123,131]]

gs = gridspec.GridSpec(5, 30)

for i, (target,frame) in enumerate(zip(targets,frames)):
    for j, f in enumerate(frame):
        img = cv2.cvtColor(cv2.imread(os.path.join(dir, target, str(f) + '.jpg')), cv2.COLOR_BGR2RGB)
        objects, _ = torch.load(os.path.join(dir, target, str(f)+'.pkl'))

        if  i in [0,1,2]:
            ax = plt.subplot(gs[i,j*6:(j+1)*6])
        else:
            ax = plt.subplot(gs[i,j*5:(j+1)*5])
        if i == 4:
            plt.imshow(img[:, 200:-200])
        else:
            plt.imshow(img)
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_xticks([])
        ax.set_yticks([])
        for object in objects:
            x_min, y_min, x_max, y_max, cls, score = object
            score = np.around(score,decimals=2)
            if i == 4:
                x_min -= 200
                x_max -= 200
            rect = patches.Rectangle((x_min, y_min), x_max - x_min,
                                 y_max - y_min, linewidth=1.5, edgecolor=color[VID_CLASSES_name[cls]], facecolor='none')
            ax.add_patch(rect)
            t = plt.text(x_min, y_min, VID_CLASSES_name[cls]+':'+str(score), fontdict=font_text)
            t.set_bbox(dict(facecolor=color[VID_CLASSES_name[cls]], alpha=0.5, edgecolor=color[VID_CLASSES_name[cls]]))

plt.show()
