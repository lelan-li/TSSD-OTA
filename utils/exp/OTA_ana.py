import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

font_label = {'family' : 'Arial',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 12,
        }

# conf=G=0.3 T=1 len=[1,10,25,50, 75, 100]
len=[1,10,25,50, 75, 100]
# mota_l = [all 29.2]
ids_l = [ 239, 235, 235, 237, 237,239] # 222
# ids_l = [ 229, 229, 231, 233, 233,233] #416
# T=1.25
# len2=[1,10,25,50, 75, 100]
# ids_l2 = [ 283, 293, 296, 299, 299,301]

# conf=G=0.3 len=10
T =      [0.5, 0.75, 1,   1.25, 1.5,]# 1.75]
mota_t = [29.2,29.2, 29.2,29.0, 28.2,]#26.5] 222
# mota_t = [29.1,29.1, 29.1,28.9, 28.2,] #416
ids_t =  [244, 244,  235, 293,  467, ]#852] 222
# ids_t =  [231, 231,  229, 275,  428, ] #416
# conf=G=0.3 len=10 T=1  6:only2, 7:only3, 8:only4, 9:only5
a = [0,1,2,3,4,5, 6,7,8,9]
ids_a_t1 = [550, 236, 236, 235, 236, 237, 240, 238, 244, 251]
# ids_a_1 = [27.9, 29.2, 29.2, 29.2, 29.2, 29.2, 29.2, 29.2]
ids_a_t125 = [550, 300, 291, 293, 292, 291, 297, 285, 290, 279]
ids_a_t05 = [550, 242, 244, 244, 249, 248, 250, 247, 258, 258]
ids416_a_t1 = [550, 234, 233, 229, 230, 229, 229, 233, 232, 222]


fig,left_axis=plt.subplots()
right_axis = left_axis.twinx()

p1, = right_axis.plot(T, mota_t, 'b-.',linewidth=3)
p2, = left_axis.plot(T, ids_t, 'r-o',linewidth=3)
plt.xlim([T[0],T[-1]])
left_axis.set_xlabel('match threshold', fontdict=font_label)
right_axis.set_ylabel('MOTA', fontdict=font_label)
left_axis.set_ylabel('IDS', fontdict=font_label)

left_axis.yaxis.label.set_color(p2.get_color())
right_axis.yaxis.label.set_color(p1.get_color())
left_axis.grid('on')
plt.legend([p2, p1], ['IDS', 'MOTA'], loc='upper left', bbox_to_anchor=(0.0,0.95), fontsize=12)

plt.show()
ax = plt.subplot(1,1,1)
plt.plot(len, ids_l, 'r-o', linewidth=3)
ax.set_ylabel('IDS', fontdict=font_label)
ax.set_xlabel('tubelet length', fontdict=font_label)
ax.grid('on')
plt.xlim([len[0],len[-1]])

plt.show()