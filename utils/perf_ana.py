import numpy as np
import matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import cv2

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

font_label = {'family' : 'Arial',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 18,
        }
font_text = {'family' : 'Arial',
        'color'  : 'w',
        'weight' : 'normal',
        'size'   : 14,
        }

font_frame = {'family' : 'Arial',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 14,
        }

ssd = {45:0.44,46:0.43,47:0.68,48:0.73,49:0.69,50:0.35,51:0.47,52:0.69,53:0.64,54:0.59,
55:0.21,56:0.27,57:0.33,58:0.53,59:0.81,60:0.91,61:0.94,62:0.91,63:0.96,64:0.86,65:0.89,
66:0.92,67:0.95,68:0.7,69:0.88,70:0.89,71:0.88,72:0.9,73:0.9,74:0.9,75:0.92,76:0.93,77:0.8,78:0.72,
79:0.75,80:0.83,81:0.61,82:0.69,83:0.87,84:0.94,85:0.96,86:0.96,87:0.97,88:0.97,89:0.96,
90:0.9,91:0.92,92:0.85,93:0.78,94:0.86,95:0.79} #frame:score

atttblstm = {45:0.92,46:0.92,47:0.95,48:0.95,49:0.96,50:0.94,51:0.93,52:0.95,53:0.94,
54:0.93,55:0.89,56:0.86,57:0.82,58:0.77,59:0.79,60:0.83,61:0.84,62:0.8,63:0.78,64:0.81,65:0.8,
66:0.83,67:0.87,68:0.84,69:0.83,70:0.79,71:0.76,72:0.76,73:0.75,74:0.75,75:0.76,76:0.76,
77:0.74,78:0.71,79:0.69,80:0.68,81:0.64,82:0.62,83:0.64,84:0.72,85:0.77,86:0.79,87:0.79,
88:0.8,89:0.8,90:0.78,91:0.79,92:0.78,93:0.76,94:0.74,95:0.71}

ssd_det = [[],[118,149,341,364],[124,93,388,371]]
tssd_det = [[143,230,306,358],[127,142,335,360],[133,100,380,374]]

print(len(ssd), len(atttblstm))
ssd_score_list = []
atttblstm_score_list = []
for frame, score in ssd.items():
    # print(score)
    ssd_score_list.append(score)

for frame, score in atttblstm.items():
    # print(score)
    atttblstm_score_list.append(score)
# print(ssd_score_list)

ssd45 = cv2.cvtColor(cv2.imread('../demo/Intro/45.jpg'), cv2.COLOR_BGR2RGB)
ssd55 = cv2.cvtColor(cv2.imread('../demo/Intro/55.jpg'), cv2.COLOR_BGR2RGB)
ssd65 = cv2.cvtColor(cv2.imread('../demo/Intro/65.jpg'), cv2.COLOR_BGR2RGB)

tssd45 = cv2.cvtColor(cv2.imread('../demo/Intro/45.jpg'), cv2.COLOR_BGR2RGB)
tssd55 = cv2.cvtColor(cv2.imread('../demo/Intro/55.jpg'), cv2.COLOR_BGR2RGB)
tssd65 = cv2.cvtColor(cv2.imread('../demo/Intro/65.jpg'), cv2.COLOR_BGR2RGB)


gs = gridspec.GridSpec(3, 3)

ax=plt.subplot(gs[0, 0])
plt.imshow(ssd45)
# t=plt.text(120, -20, 'frame:45',fontdict=font_frame)
# t.set_bbox(dict(facecolor='orange', alpha=0.5, edgecolor='orange'))
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_xticks([])
ax.set_yticks([])
plt.title('frame:45', fontdict=font_label)
plt.ylabel('SSD', fontdict=font_label)
ax=plt.subplot(gs[0, 1])
plt.imshow(ssd55)
rect = patches.Rectangle((ssd_det[1][0], ssd_det[1][1],),ssd_det[1][2]-ssd_det[1][0],ssd_det[1][3]-ssd_det[1][1],linewidth=1.5,edgecolor='c',facecolor='none')
ax.add_patch(rect)
t=plt.text(ssd_det[1][0], ssd_det[1][1], 'rabbit',fontdict=font_text)
t.set_bbox(dict(facecolor='c', alpha=0.5, edgecolor='c'))
ax.spines['left'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_xticks([])
ax.set_yticks([])
plt.title('frame:55', fontdict=font_label)
ax=plt.subplot(gs[0, 2])
plt.imshow(ssd65)
rect = patches.Rectangle((ssd_det[2][0], ssd_det[2][1],),ssd_det[2][2]-ssd_det[2][0],ssd_det[2][3]-ssd_det[2][1],linewidth=1.5,edgecolor='r',facecolor='none')
ax.add_patch(rect)
t=plt.text(ssd_det[2][0], ssd_det[2][1], 'hamster',fontdict=font_text)
t.set_bbox(dict(facecolor='r', alpha=0.5, edgecolor='r'))
ax.spines['bottom'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_xticks([])
ax.set_yticks([])
plt.title('frame:65', fontdict=font_label)
ax=plt.subplot(gs[1, 0])
plt.imshow(tssd45)
rect = patches.Rectangle((tssd_det[0][0], tssd_det[0][1],),tssd_det[0][2]-tssd_det[0][0],tssd_det[0][3]-tssd_det[0][1],linewidth=1.5,edgecolor='r',facecolor='none')
ax.add_patch(rect)
t=plt.text(tssd_det[0][0], tssd_det[0][1], 'hamster',fontdict=font_text)
t.set_bbox(dict(facecolor='r', alpha=0.5, edgecolor='r'))
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_xticks([])
ax.set_yticks([])
plt.ylabel('TSSD', fontdict=font_label)
ax=plt.subplot(gs[1, 1])
plt.imshow(tssd55)
rect = patches.Rectangle((tssd_det[1][0], tssd_det[1][1],),tssd_det[1][2]-tssd_det[1][0],tssd_det[1][3]-tssd_det[1][1],linewidth=1.5,edgecolor='r',facecolor='none')
ax.add_patch(rect)
t=plt.text(tssd_det[1][0], tssd_det[1][1], 'hamster',fontdict=font_text)
t.set_bbox(dict(facecolor='r', alpha=0.5, edgecolor='r'))
ax.spines['bottom'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_xticks([])
ax.set_yticks([])
ax=plt.subplot(gs[1, 2])
plt.imshow(tssd65)
rect = patches.Rectangle((tssd_det[2][0], tssd_det[2][1],),tssd_det[2][2]-tssd_det[2][0],tssd_det[2][3]-tssd_det[2][1],linewidth=1.5,edgecolor='r',facecolor='none')
ax.add_patch(rect)
t=plt.text(tssd_det[2][0], tssd_det[2][1], 'hamster',fontdict=font_text)
t.set_bbox(dict(facecolor='r', alpha=0.5, edgecolor='r'))
ax.spines['bottom'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_xticks([])
ax.set_yticks([])

frame = range(45,96)
ax=plt.subplot(gs[2, :])
ssd_plt, = plt.plot(frame,ssd_score_list,color='c', linewidth=2.0)
tssd_plt, = plt.plot(frame,atttblstm_score_list,color='red', linewidth=2.0)
plt.legend([ssd_plt, tssd_plt], ['SSD', 'TSSD'], loc='lower right',fontsize=12)
plt.xlim(1,len(ssd))
plt.ylabel('score',fontdict=font_label)
plt.xlabel('frame',fontdict=font_label)
plt.xlim([45,96])
plt.grid('on')
plt.show()