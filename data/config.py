# config.py
import os.path
from data import VOC_CLASSES, VID_CLASSES, UW_CLASSES

# gets home dir cross platform
home = os.path.expanduser("~")
vocdir = os.path.join(home,"data/VOCdevkit/")
viddir = os.path.join(home,"data/ILSVRC/")
mot17detdir = os.path.join(home,"data/MOT/MOT17Det/")
mot15dir = os.path.join(home,"data/MOT/2DMOT2015/")
uwdir = os.path.join(home,"data/UWdevkit/")

# note: if you used our download scripts, this should be right
VOCroot = vocdir # path to VOCdevkit root dir
VIDroot = viddir
MOT17Detroot = mot17detdir
MOT15root = mot15dir
UWroot = uwdir
COCOroot = os.path.join(home,"data/MSCOCO2017/")

# default batch size
BATCHES = 32
# data reshuffled at every epoch
SHUFFLE = True
# number of subprocesses to use for data loading
WORKERS = 4

dataset_training_cfg = {'VOC0712':([('2007', 'trainval'), ('2012', 'trainval')], len(VOC_CLASSES) + 1, VOCroot),
                        'VIDDET': ('train', len(VID_CLASSES) + 1, VIDroot),
                        'VID2017': ('train', len(VID_CLASSES) + 1, VIDroot),
                        'seqVID2017': ('train_remove_noobject', len(VID_CLASSES) + 1, VIDroot),
                        'MOT17Det': ('train', 2, MOT17Detroot),
                        'seqMOT17Det': ('train_video', 2, MOT17Detroot),
                        'MOT15': ('train15_17', 2, MOT15root),
                        'seqMOT15': ('train_video', 2, MOT15root),
                        'UW': ('train', len(UW_CLASSES) + 1, UWroot),
                        'seqUW': ('train', len(UW_CLASSES) + 1, UWroot),
                        }

#SSD300 CONFIGS
# newer version: use additional conv11_2 layer as last layer before multibox layers
VOC_300 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim' : 300,

    'steps' : [8, 16, 32, 64, 100, 300],

    'min_sizes' : [30, 60, 111, 162, 213, 264],

    'max_sizes' : [60, 111, 162, 213, 264, 315],

    # 'aspect_ratios' : [[2, 1/2], [2, 1/2, 3, 1/3], [2, 1/2, 3, 1/3],
    #                    [2, 1/2, 3, 1/3], [2, 1/2], [2, 1/2]],
    'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    # 'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2,3], [2,3]],

    'variance' : [0.1, 0.2],

    'clip' : True,

    'flip': True,

    'name' : 'VOC_300',
}

VOC_320 = {
    'feature_maps': [40, 20, 10, 5],

    'min_dim': 320,

    'steps': [8, 16, 32, 64],

    'min_sizes': [32, 64, 128, 256],

    'max_sizes': [],

    'aspect_ratios': [[2], [2], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,

    'flip': True,

    'name': 'VOC_320',
}

VOC_512_RefineDet = {
    'feature_maps': [64, 32, 16, 8],

    'min_dim': 512,

    'steps': [8, 16, 32, 64],

    'min_sizes': [32, 64, 128, 256],

    'max_sizes': [],

    'aspect_ratios': [[2], [2], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,

    'flip': True,

    'name': 'VOC_512_RefineDet',
}


VOC_512= {
    'feature_maps' : [64, 32, 16, 8, 4, 2, 1],

    'min_dim' : 512,

    'steps' : [8, 16, 32, 64, 128, 256, 512],

    'min_sizes'  : [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8 ],

    'max_sizes'  : [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],

    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2,3], [2], [2]],

    'variance' : [0.1, 0.2],

    'clip' : True,

    'flip': True,

    'name' : 'VOC_512'
}

MOT_300 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim' : 300,

    'steps' : [8, 16, 32, 64, 100, 300],

    'min_sizes' : [30, 60, 111, 162, 213, 264],

    'max_sizes' : [60, 111, 162, 213, 264, 315],

    # 'aspect_ratios': [[1 / 2, 1 / 3], [1 / 2, 1 / 3], [1 / 2, 1 / 3], [1 / 2, 1 / 3],
    #                   [1 / 2, 1 / 3], [1 / 2, 1 / 3]],

    # 'aspect_ratios' : [[2,3,4], [2, 3,4], [2, 3, 4], [2, 3, 4], [2,3,4], [2,3,4]],
    'aspect_ratios': [[1 / 2, 1 / 3, 1/4], [1 / 2, 1 / 3, 1/4], [1 / 2, 1 / 3, 1/4], [1 / 2, 1 / 3, 1/4], [1 / 2, 1 / 3, 1/4], [1 / 2, 1 / 3, 1/4]],

    'variance' : [0.1, 0.2],

    'clip' : True,

    'flip': False,

    'name' : 'MOT_300',
}

COCO_300 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim' : 300,

    'steps' : [8, 16, 32, 64, 100, 300],

    'min_sizes' : [21, 45, 99, 153, 207, 261],

    'max_sizes' : [45, 99, 153, 207, 261, 315],

    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance' : [0.1, 0.2],

    'clip' : True,

    'flip': True,

    'name': 'COCO_300'
}

COCO_512= {
    'feature_maps' : [64, 32, 16, 8, 4, 2, 1],

    'min_dim' : 512,

    'steps' : [8, 16, 32, 64, 128, 256, 512],

    'min_sizes' : [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],

    'max_sizes' : [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],

    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2,3], [2], [2]],

    'variance' : [0.1, 0.2],

    'flip': True,

    'clip' : True,

    'name': 'COCO_512'
}

mb_cfg = {'VOC_300':VOC_300, 'VOC_320':VOC_320, 'VOC_512':VOC_512, 'MOT_300':MOT_300,
          'COCO_300':COCO_300, 'COCO_512':COCO_512, 'VOC_512_RefineDet': VOC_512_RefineDet}
