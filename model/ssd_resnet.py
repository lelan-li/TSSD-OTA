import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from data import VOC_300, VOC_512
import os
from .networks import ConvAttention, Bottleneck, PreModule

class SSD_ResNet(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, res_layers, extra_cfg, mb_cfg, num_classes, size=300, top_k=200, thresh=0.01, nms_thresh=0.45, prior='VOC_ResNet_300', pm=0., device= torch.device('cpu')):
        super(SSD_ResNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.device = device
        self.pm_flag= (True, False)[pm == 0.0]
        if prior=='VOC_300':
            self.priorbox = PriorBox(VOC_300)
        elif prior=='VOC_512':
            self.priorbox = PriorBox(VOC_512)
        else:
            print('Unkown prior type')

        with torch.no_grad():
            self.priors = self.priorbox.forward().to(self.device)
        self.size = size

        # ResNet
        self.inplanes = 64
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU(inplace=True)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layer1 = self._make_layer(Bottleneck, 64, res_layers[0])
        layer2 = self._make_layer(Bottleneck, 128, res_layers[1], stride=2)
        layer3 = self._make_layer(Bottleneck, 256, res_layers[2], stride=2)
        layer4 = self._make_layer(Bottleneck, 512, res_layers[3], stride=1)
        avgpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv5_pre = nn.Conv2d(2048, 1024, kernel_size=1, bias=False)
        bn5_pre = nn.BatchNorm2d(conv5_pre.out_channels)
        conv5 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False)
        bn5 = nn.BatchNorm2d(conv5.out_channels)
        self.backbone = nn.ModuleList([conv1, bn1, relu, maxpool, layer1, layer2, layer3,
                                     layer4, avgpool, conv5_pre, bn5_pre,relu, conv5, bn5, relu])
        # extra
        extra_layers = []
        flag = False
        in_channels = conv5.out_channels
        for k, v in enumerate(extra_cfg):
            if in_channels != 'S':
                if v == 'S':
                    extra_layers.append( nn.Sequential(nn.Conv2d(in_channels, extra_cfg[k + 1],
                                                             kernel_size=(1, 3)[flag], stride=2, padding=1, bias=False),
                                                   nn.BatchNorm2d(extra_cfg[k + 1]), nn.ReLU(inplace=True)))
                else:
                    extra_layers.append(
                        nn.Sequential(nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag], bias=False), nn.BatchNorm2d(v),
                                      nn.ReLU(inplace=True)))
                flag = not flag
            in_channels = v
        if self.size == 512:
            extra_layers.append(nn.Sequential(nn.Conv2d(in_channels, int(extra_cfg[-1]/2), kernel_size=1, stride=1), nn.BatchNorm2d(int(extra_cfg[-1]/2)),
                                      nn.ReLU(inplace=True)))
            extra_layers.append(nn.Sequential(nn.Conv2d(int(extra_cfg[-1]/2), extra_cfg[-1], kernel_size=4, stride=1, padding=1), nn.BatchNorm2d(extra_cfg[-1]),
                                      nn.ReLU(inplace=True)))
        self.extras = nn.ModuleList(extra_layers)
        # prediction model
        out_channels = [layer2[-1].conv3.out_channels, conv5.out_channels,
                        extra_layers[1][0].out_channels, extra_layers[3][0].out_channels,
                        extra_layers[5][0].out_channels, extra_layers[7][0].out_channels]
        if self.size == 512:
            out_channels.append(extra_layers[9][0].out_channels)
        if self.pm_flag:
            pm_list = []
            for i, oc in enumerate(out_channels):
                pm_list += [PreModule(oc, pm)]
                out_channels[i] = int(oc*pm)
            self.pm = nn.ModuleList(pm_list)
        # multi_box
        loc_layers = []
        conf_layers = []

        for o, v in zip(out_channels, mb_cfg):
            loc_layers += [nn.Conv2d(o, v * 4, kernel_size=3, padding=1, bias=False)]
            conf_layers += [nn.Conv2d(o, v * num_classes, kernel_size=3, padding=1, bias=False)]
        self.loc = nn.ModuleList(loc_layers)
        self.conf = nn.ModuleList(conf_layers)


        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)

        if phase == 'test':
            self.softmax = nn.Softmax(dim=1)
            self.detect = Detect(num_classes, 0, top_k=top_k, conf_thresh=thresh, nms_thresh=nms_thresh)
                                # num_classes, bkg_label, top_k, conf_thresh, nms_thresh
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        a_map = list()

        for i in range(6):
            x = self.backbone[i](x)
        s = self.L2Norm(x)
        sources.append(s)
        for i in range(6, len(self.backbone)):
            x = self.backbone[i](x)
        sources.append(x)

        for k, v in enumerate(self.extras):
            x = v(x)
            if k % 2 == 1:
                sources.append(x)
        if self.pm_flag:
            for i, p in  enumerate(self.pm):
                sources[i] = p(sources[i])

        for (f, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(f).permute(0, 2, 3, 1).contiguous()) # [ith_multi_layer, batch, height, width, out_channel]
            conf.append(c(f).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                self.priors,                 # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors,
            )
        return output, tuple(a_map)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


mbox = {
    'VOC_300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    'VOC_512': [6, 6, 6, 6, 6, 4, 4],
}
extras = {
    'ResNet300': [256, 'S', 512, 256, 'S', 512, 256, 512, 256, 512],
    'ResNet512': [256, 'S', 512, 256, 'S', 512, 256, 'S', 512, 256, 'S', 512],
}

def build_ssd_resnet(phase, backbone='ResNet18', size=300, num_classes=21, top_k=200, thresh=0.01, prior='VOC_300',
              nms_thresh=0.45, pm=0., device=torch.device('cpu')):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size not in [300, 512]:
        print("Error: Sorry only SSD300/512 is supported currently!")
        return
    if backbone == 'ResNet18':
        res_layers = [2,2,2,2]
    elif backbone == 'ResNet34':
        res_layers = [2,2,2,2]
    elif backbone == 'ResNet50':
        res_layers = [3, 4, 6, 3]
    elif backbone == 'ResNet101':
        res_layers = [3, 4, 22, 3]
    else:
        print("Error: Unknown model!")
        return

    return SSD_ResNet(phase,res_layers, extras['ResNet'+str(size)], mbox[prior], num_classes,size=size,
                   top_k=top_k,thresh= thresh,nms_thresh=nms_thresh, prior=prior, pm=pm, device=device)

