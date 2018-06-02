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

color_id = ['r','g','b','m', 'orange']

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
dir = '../../demo/OTA'

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

targets = ['ETH-Jelmoli', 'ETH-Linthescher', 'PETS09-S2L2']
frames = [[318, 332,355,376,392], [643,756,1006,1065, 1140],[45,55,67,83,96]]
key_id = [[56], [149], [0,3,4,23,30]]
gs = gridspec.GridSpec(len(targets), len(frames[0]))

for i, (target,frame) in enumerate(zip(targets,frames)):

    for j, f in enumerate(frame):
        img = cv2.cvtColor(cv2.imread(os.path.join(dir, target, str(f) + '.jpg')), cv2.COLOR_BGR2RGB)
        objects = torch.load(os.path.join(dir, target, 'tblstm_'+str(f)+'.pkl'))

        ax = plt.subplot(gs[i, j])
        plt.imshow(img)
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_xticks([])
        ax.set_yticks([])

        for object in objects:
            x_min, y_min, x_max, y_max, cls, score, id = object
            score = np.around(score,decimals=2)
            if id in key_id[i]:
                # print(key_id[i].index(id))
                color = color_id[key_id[i].index(id)]
            else:
                color = 'gray'
            rect = patches.Rectangle((x_min, y_min), x_max - x_min,
                                 y_max - y_min, linewidth=1.0, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            t = plt.text(x_min, y_min, str(int(id)), color='w', fontdict=font_text)
            t.set_bbox(dict(facecolor=color, alpha=0.5, edgecolor=color))
            if(j==0):
                plt.title('frame:'+str(f), fontdict=font_label)
            else:
                plt.title(str(f), fontdict=font_label)

plt.show()