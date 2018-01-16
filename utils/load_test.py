import torch
from ssd import build_ssd
from collections import OrderedDict

ssd_net = build_ssd('train', 300, 31, tssd='conf_conv_lstm')
net = torch.nn.DataParallel(ssd_net)
torch.save(net.state_dict(),'test.pth')
vgg_weights = torch.load('../weights/vgg16_reducedfc.pth')

ssd_weights = torch.load('../weights/ssd300_VID2017/ssd300_VID2017_290000.pth')
ssd_vgg_weights = OrderedDict()
ssd_extras_weights = OrderedDict()
for key, weight in ssd_weights.items():
    key_split = key.split('.')
    subnet_name = key_split[0]
    if subnet_name == 'vgg':
        ssd_vgg_weights[key_split[1]+'.'+key_split[2]] = weight
    elif subnet_name == 'extras':
        ssd_extras_weights[key_split[1]+'.'+key_split[2]] = weight

# ssd_net.load_weights('../weights/ssd300_VID2017/ssd300_VID2017_290000.pth')
print(type(ssd_net.parameters()))
for param in ssd_net.parameters():
    print(param.key)


# vgg_dict = {k:v for k,v in weights}
# ssd_net.vgg.load_state_dict(weights.vgg)
# ssd_net.extras.load_state_dict(weights.extras)
