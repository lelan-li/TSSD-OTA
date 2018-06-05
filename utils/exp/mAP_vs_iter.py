import matplotlib
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

eval_dir = '../../eval'
model_name = ['ssd320RefineFalse_VOCb32', 'ssd512RefineFalse_VOCb32']
dataset_name = 'VOC0712'
val_type = 'test'
plot_iter = list(range(5000, 120000, 5000))
# plot_iter = [ 5000, 10000, 15000, 20000, 30000, 40000, 50000]
# plot_iter = [ 10000, 20000, 30000, 40000, 50000, 60000, 70000,  80000]
curve_list = []

for name in model_name:
    mAP_dir = join(eval_dir, name)
    mAP_list = []
    for iter in plot_iter:
        sub_dir = join(mAP_dir, str(iter)+'_'+dataset_name+'_'+val_type)
        for f in listdir(sub_dir):
            if isfile(join(sub_dir, f)) and f.split('.')[-1]=='txt':
                mAP_list.append(float(f.split('.')[0])/100)
                break
    max_mAP = max(mAP_list)
    max_iter = mAP_list.index(max_mAP)
    curve, = plt.plot(plot_iter, mAP_list)
    plt.scatter(plot_iter[max_iter], max_mAP)
    curve_list.append(curve)
    print(name, ': ', max_mAP, plot_iter[max_iter])
plt.legend(curve_list, model_name, loc='lower right',fontsize=12)
plt.grid('on')
plt.show()
