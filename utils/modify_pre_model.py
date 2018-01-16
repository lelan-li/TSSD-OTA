import torch
from collections import OrderedDict

pre_weights_name = 'weights/tssd300_VID2017_b2_s32_SkipShare_preVggExtraLocConf_bycicle/ssd300_seqVID2017_10000.pth'
pre_weights = torch.load('../'+pre_weights_name)
new_weights = OrderedDict()

for key, weight in pre_weights.items():
    key_split = key.split('.')
    subnet_name = key_split[0]
    if subnet_name == 'conf_lstm':
        key = key[5:]
    new_weights[key] = weight

new_weights_name = '../' + pre_weights_name.split('.')[0]+'_modi.pth'
print(new_weights_name)
torch.save(new_weights, new_weights_name)
