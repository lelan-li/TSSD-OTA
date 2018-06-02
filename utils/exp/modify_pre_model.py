import torch
from collections import OrderedDict
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

pre_weights_name = 'weights040/resnet50-19c8e357.pth'
# pre_weights_name = 'weights/tssd300_VID2017_b8s8_RSkipTBDoLstm_Drop2Clip5_FixVggExtraPreLocConf/ssd300_seqVID2017_20000.pth'
pre_weights = torch.load('../../'+pre_weights_name)
new_weights = pre_weights
del new_weights['fc.weight']
del new_weights['fc.bias']
# new_weights['conv5.bias'] = new_weights['conv1.bias']

upsample = torch.nn.Upsample(size=(1024,2048,3,3))
new_weights['conv5_pre.weight'] = new_weights['layer4.2.conv1.weight'].repeat(2,1,1,1)
new_weights['bn5_pre.running_mean'] =  new_weights['layer4.2.bn2.running_mean'].repeat(2)
new_weights['bn5_pre.running_var'] =  new_weights['layer4.2.bn2.running_var'].repeat(2)
new_weights['bn5_pre.weight'] =  new_weights['layer4.2.bn2.weight'].repeat(2)
new_weights['bn5_pre.bias'] =  new_weights['layer4.2.bn2.bias'].repeat(2)

new_weights['conv5.weight'] = new_weights['layer4.2.conv2.weight'].repeat(2,2,1,1)
new_weights['bn5.running_mean'] =  new_weights['layer4.2.bn2.running_mean'].repeat(2)
new_weights['bn5.running_var'] =  new_weights['layer4.2.bn2.running_var'].repeat(2)
new_weights['bn5.weight'] =  new_weights['layer4.2.bn2.weight'].repeat(2)
new_weights['bn5.bias'] =  new_weights['layer4.2.bn2.bias'].repeat(2)


new_weights_rename = OrderedDict()
res_layer = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool','conv5_pre', 'bn5_pre','relu', 'conv5', 'bn5', 'relu']
for key, weight in new_weights.items():
    key_split = key.split('.')
    layer_name = key_split[0]
    subnet_name = ''
    for i in key_split[1:]:
        subnet_name += i+'.'
    subnet_name = subnet_name[:-1]
    re_name = str(res_layer.index(layer_name)) + '.'+subnet_name
    print(re_name)
    new_weights_rename[re_name] = weight
    # if key in ['vgg.33.weight']:
    #     weight = weight.resize_(512,1024,1,1)
    # elif key in ['vgg.33.bias']:
    #     weight = weight.resize_(512)
        # print(weight.size())
    # elif key in ['extras.0.weight']:
    #     weight = weight.resize_(256,512,1,1)
    # elif key in ['extra.0.bias']:
    #     weight = weight.resize_(24,256,3,3)
    # elif key in ['loc.0.bias',  'loc.4.bias','loc.5.bias']:
    #     weight = weight.resize_(24)
    # elif key in ['conf.0.bias','conf.5.bias','conf.4.bias']:
    #     weight = weight.resize_(186)

    # new_weights[key] = weight

new_weights_name = '../../weights040/resnet50_reducefc.pth'
print(new_weights_name)
torch.save(new_weights_rename, new_weights_name)
