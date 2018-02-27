import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import torch
import os
import cv2

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

dir = '../demo/res'
targets = ['airplane7011', 'bird61000']
frames = ['ssd_1', 'ssd_20', '1', '20']
label = ['', '(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
title = ['frame:1', 'frame:20']

font_label = {'family' : 'Arial',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 12,
        }

gs = gridspec.GridSpec(9, 12)

att_list = []
res_list = []
for target_index, target in enumerate(targets):
    for frame_index, frame in enumerate(frames):
        # img_list.append(cv2.cvtColor(cv2.imread(os.path.join(dir, str(frame)+'.jpg')),cv2.COLOR_BGR2RGB))
        if frame in ['1', '20']:
            # print(target_index, frame_index, target_index*6+(frame_index-2)*3)
            img = cv2.cvtColor(cv2.imread(os.path.join(dir, target, frame + '.jpg')), cv2.COLOR_BGR2RGB)
            if frame == '1':
                ax = plt.subplot(gs[0,target_index*6+(frame_index-2)*3:target_index*6+(frame_index-2)*3+3])
            else:
                ax = plt.subplot(gs[0,target_index*6+(frame_index-2)*3:target_index*6+(frame_index-2)*3+3])
            plt.imshow(img[:,200:-200])
            ax.spines['right'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.title(title[frame_index-2], fontdict=font_label)

        objects, att_map = torch.load(os.path.join(dir, target, frame+'.pkl'))
        att = [att[0][0].cpu().data.numpy()[:,200:-200] for att in att_map]
        # for scale,att in enumerate(att_map):
        #     if scale == 0:
        #         att0 = att
        #     else:
        att_cat = np.concatenate(att, axis=1)

        ax = plt.subplot(gs[target_index*4+frame_index+1,:])
        hot = plt.imshow(att_cat, cmap=plt.get_cmap('jet'))
        # plt.colorbar(hot, cax=None, ax=None, shrink=0.2)  # 长度为半
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.ylabel(label[target_index*4+frame_index+1], fontdict=font_label)

plt.show()
    # pass