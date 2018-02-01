import torch
from collections import OrderedDict
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# pre_weights_name = 'weights/vgg16_reducedfc.pth'
pre_weights_name = 'weights/ssd300_VIDDET/ssd300_VIDDET_160000_512.pth'
pre_weights = torch.load('../'+pre_weights_name)
new_weights = OrderedDict()
upsample = torch.nn.Upsample()

for key, weight in pre_weights.items():
    # key_split = key.split('.')
    # subnet_name = key_split[0]+'.'+key_split[1]+'.'+key_split[2]
    # print(subnet_name)
    if key in ['vgg.33.weight']:
        weight = weight.resize_(512,1024,1,1)
    elif key in ['vgg.33.bias']:
        weight = weight.resize_(512)
    #     # print(weight.size())
    elif key in ['extras.0.weight']:
        weight = weight.resize_(256,512,1,1)
    # elif key in ['extra.0.bias']:
    #     weight = weight.resize_(24,256,3,3)
    # elif key in ['loc.0.bias',  'loc.4.bias','loc.5.bias']:
    #     weight = weight.resize_(24)
    # elif key in ['conf.0.bias','conf.5.bias','conf.4.bias']:
    #     weight = weight.resize_(186)

    new_weights[key] = weight

new_weights_name = '../' + pre_weights_name.split('.')[0]+'_new.pth'
print(new_weights_name)
torch.save(new_weights, new_weights_name)
