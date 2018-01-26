import torch
from collections import OrderedDict
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

pre_weights_name = 'weights/ssd300_VIDDET/ssd300_VIDDET_160000_512.pth'
pre_weights = torch.load('../'+pre_weights_name)
new_weights = OrderedDict()
upsample = torch.nn.Upsample()

for key, weight in pre_weights.items():
    # key_split = key.split('.')
    # subnet_name = key_split[0]+'.'+key_split[1]+'.'+key_split[2]
    # print(subnet_name)
    if key in ['conf.1.weight', 'loc.1.weight']:
        size = weight.size()
        # print(size)
        weight = weight[:,0:size[1]:2,:,:]
        # print(weight.size())
    new_weights[key] = weight

new_weights_name = '../' + pre_weights_name.split('.')[0]+'_512.pth'
print(new_weights_name)
torch.save(new_weights, new_weights_name)
