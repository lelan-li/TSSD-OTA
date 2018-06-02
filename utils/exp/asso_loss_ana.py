import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True

mAP = [65.30, 65.43, 65.16, 65.19, 65.05, 65.09, 65.13]
theta = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

font_label = {'family' : 'Arial',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 12,
        }

plt.plot(theta, mAP, '-bo', linewidth=3)
plt.xlim([0, 1])
plt.ylabel('mAP(%)', fontdict=font_label)
plt.xlabel(r'$\theta$', fontdict=font_label)
plt.grid('on')
plt.show()