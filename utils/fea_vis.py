import numpy as np
import matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import torch
import os
import cv2

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

font_label = {'family' : 'Arial',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 12,
        }

feature_path = '../demo/res/car11005'
feature_path_2 = '../demo/res/car11005_withoutAtt'

frame_list = ['60','80','100']

gs = gridspec.GridSpec(5, len(frame_list))

for idx, frame in enumerate(frame_list):
    im = cv2.imread(os.path.join(feature_path, frame + '.jpg'))
    _, state,_ = torch.load(os.path.join(feature_path, frame + '.pkl'))
    _, state2 = torch.load(os.path.join(feature_path_2, frame + '.pkl'))
    cell, hidden = state[2]
    cell2, hidden2 = state2[2]
    # hidden = cell
    cell_map = torch.zeros(cell.size()[2:])
    hidden_map = torch.zeros(hidden.size()[2:])
    cell2_map = torch.zeros(cell2.size()[2:])
    hidden2_map = torch.zeros(hidden2.size()[2:])

    for i in range(cell_map.size(0)):
        for j in range(cell_map.size(1)):
            # a = cell.cpu().data[0,:,i,j]
            cell_map[i,j] = torch.norm(cell.cpu().data[0,:,i,j], 2)
            hidden_map[i, j] = torch.norm(hidden.cpu().data[0, :, i, j], 2)
            cell2_map[i, j] = torch.norm(cell2.cpu().data[0, :, i, j], 2)
            hidden2_map[i, j] = torch.norm(hidden2.cpu().data[0, :, i, j], 2)

    # cell_max_norm = torch.max(cell_map)
    # cell_map /= cell_max_norm
    #
    # hidden_max_norm = torch.max(hidden_map)
    # hidden_map /= hidden_max_norm
    cell_map = cv2.resize(cell_map.numpy(), (im.shape[1], im.shape[0]))
    cell2_map = cv2.resize(cell2_map.numpy(), (im.shape[1], im.shape[0]))
    hidden_map = cv2.resize(hidden_map.numpy(), (im.shape[1], im.shape[0]))
    hidden2_map = cv2.resize(hidden2_map.numpy(), (im.shape[1], im.shape[0]))

    ax = plt.subplot(gs[0, idx])
    plt.imshow(im)
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title('frame: '+frame, fontdict=font_label)

    ax = plt.subplot(gs[1, idx])
    plt.imshow(cell2_map, cmap=plt.get_cmap('jet'))
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])
    if idx == 0:
        plt.ylabel('(a)', fontdict=font_label)

    ax = plt.subplot(gs[2, idx])
    plt.imshow(cell_map, cmap=plt.get_cmap('jet'))
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])
    if idx == 0:
        plt.ylabel('(b)', fontdict=font_label)

    ax = plt.subplot(gs[3, idx])
    plt.imshow(hidden2_map, cmap=plt.get_cmap('jet'))
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])
    if idx == 0:
        plt.ylabel('(c)', fontdict=font_label)

    ax = plt.subplot(gs[4, idx])
    plt.imshow(hidden_map, cmap=plt.get_cmap('jet'))
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])
    if idx == 0:
        plt.ylabel('(d)', fontdict=font_label)

plt.show()


