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

font_text = {'family' : 'Arial',
        'color'  : 'w',
        'weight' : 'normal',
        'size'   : 8,
        }

color_cls = ['r','b','orange']
gs = gridspec.GridSpec(2, 6)
video = '../../demo/OTA/2_GAN_RS_040'
frames = [8,71,99,274,499,524,610,798,964,1114,1259,1400]
# axes = [2,3,4,5,8,9,10,11,12,13,14,15]

# rov = cv2.cvtColor(cv2.imread(os.path.join(video, 'rov.jpg'))[0:-100,500:-600], cv2.COLOR_BGR2RGB)
# ax = plt.subplot(gs[0])
# plt.imshow(rov)
# ax.spines['right'].set_color('none')
# ax.spines['left'].set_color('none')
# ax.spines['bottom'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.set_xticks([])
# ax.set_yticks([])
# grasp = cv2.cvtColor(cv2.imread(os.path.join(video, 'grasp.jpg'))[0:-100,500:-600], cv2.COLOR_BGR2RGB)
# ax = plt.subplot(gs[-1])
# plt.imshow(grasp)
# ax.spines['right'].set_color('none')
# ax.spines['left'].set_color('none')
# ax.spines['bottom'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.set_xticks([])
# ax.set_yticks([])
for i, f in enumerate(frames):
    img = cv2.cvtColor(cv2.imread(os.path.join(video, str(f) + '.jpg')), cv2.COLOR_BGR2RGB)
    objects = torch.load(os.path.join(video, 'tblstm_' + str(f) + '.pkl'))
    ax = plt.subplot(gs[i])
    plt.imshow(img)
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])
    for object in objects:
        x_min, y_min, x_max, y_max, cls, score, id = object
        score = np.around(score, decimals=2)
        if not isinstance(cls, int):
            print(cls)
        color = color_cls[cls]
        rect = patches.Rectangle((x_min, y_min), x_max - x_min,
                                 y_max - y_min, linewidth=1.0, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        t = plt.text(x_min, y_min, str(int(id)), color=color,
                     fontdict=font_text)
        t.set_bbox(dict(facecolor='w', alpha=0.2, edgecolor=color))

plt.show()