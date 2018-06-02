import matplotlib
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

eval_dir = '../../eval'
model_name = 'ssd320RefineFalse_UW'
dataset_name = 'UW'
val_type = 'val'
mAP_dir = join(eval_dir, model_name)
plot_iter = [ 5000, 10000, 15000, 20000, 30000, 40000, 50000]
# plot_iter = [ 10000, 20000, 30000, 40000, 50000, 60000, 70000,  80000]
mAP_list = []
for iter in plot_iter:
    sub_dir = join(mAP_dir, str(iter)+'_'+dataset_name+'_'+val_type)
    for f in listdir(sub_dir):
        if isfile(join(sub_dir, f)) and f.split('.')[-1]=='txt':
            mAP_list.append(float(f.split('.')[0])/100)
            break

plt.plot(plot_iter, mAP_list)
plt.show()