import torch
from model import build_ssd, build_ssd_resnet
from collections import OrderedDict

ssdbn_net = build_ssd('train', 300, 31, bn=False)
ssd_net = build_ssd('train', 300, 31, bn=False)
# net = torch.nn.DataParallel(ssd_net)
# torch.save(net.state_dict(),'test.pth')
vgg_weights = torch.load('../../weights/vgg16_reducedfc_512.pth')
vggbn_weights = torch.load('../../weights/vgg16_bn-6c64b313.pth')

# ssd_weights = torch.load('../weights/ssd300_VID2017/ssd300_VID2017_290000.pth')
vgg16bn_reducedfc_512_weights = OrderedDict()
# ssd_extras_weights = OrderedDict()
for key, weight in vggbn_weights.items():
    key_class = key.split('.')[0]
    if key_class == 'features':
        key = key.split('.')[1] + '.' + key.split('.')[2]
        vgg16bn_reducedfc_512_weights[key] = weight

vgg16bn_reducedfc_512_weights['44.weight'] = vgg_weights['31.weight']
vgg16bn_reducedfc_512_weights['44.bias'] = vgg_weights['31.bias']

vgg16bn_reducedfc_512_weights['45.weight'] = vgg16bn_reducedfc_512_weights['41.weight'].repeat(2)
vgg16bn_reducedfc_512_weights['45.bias'] = vgg16bn_reducedfc_512_weights['41.bias'].repeat(2)
vgg16bn_reducedfc_512_weights['45.running_mean'] = vgg16bn_reducedfc_512_weights['41.running_mean'].repeat(2)
vgg16bn_reducedfc_512_weights['45.running_var'] = vgg16bn_reducedfc_512_weights['41.running_var'].repeat(2)

vgg16bn_reducedfc_512_weights['47.weight'] = vgg_weights['33.weight']
vgg16bn_reducedfc_512_weights['47.bias'] = vgg_weights['33.bias']

vgg16bn_reducedfc_512_weights['48.weight'] = vgg16bn_reducedfc_512_weights['41.weight']
vgg16bn_reducedfc_512_weights['48.bias'] = vgg16bn_reducedfc_512_weights['41.bias']
vgg16bn_reducedfc_512_weights['48.running_mean'] = vgg16bn_reducedfc_512_weights['41.running_mean']
vgg16bn_reducedfc_512_weights['48.running_var'] = vgg16bn_reducedfc_512_weights['41.running_var']


# torch.save(vgg16bn_reducedfc_512_weights, '../../weights/vgg16allbn_reducedfc_512.pth')


