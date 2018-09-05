import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import os
from ssd import mbox, extras, ConvAttention, ConvLSTMCell
from data import v2

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SSD_ResNet(nn.Module):

    def __init__(self, phase, backbone, res_layers, extra_cfg, mb_cfg, num_classes, size=300, c7_channel=1024, top_k=200, thresh=0.01, nms_thresh=0.45):
        super(SSD_ResNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.priorbox = PriorBox(v2)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = 300
        self.size = size

        # ResNet
        self.inplanes = 64
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU(inplace=True)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        avgpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        if backbone == 'ResNet18':
            layer1 = self._make_layer(BasicBlock, 64, res_layers[0])
            layer2 = self._make_layer(BasicBlock, 128, res_layers[1], stride=1)
            layer3 = self._make_layer(BasicBlock, 256, res_layers[2], stride=1)
            layer4 = self._make_layer(BasicBlock, 512, res_layers[3], stride=2)
            conv5_pre = nn.Conv2d(512, 1024, kernel_size=1, bias=False)
            conv5 = nn.Conv2d(1024, c7_channel, kernel_size=3, stride=2, padding=1, bias=False)
            self.conv4_3_layer = 8
        else:
            layer1 = self._make_layer(Bottleneck, 64, res_layers[0])
            layer2 = self._make_layer(Bottleneck, 128, res_layers[1], stride=2)
            layer3 = self._make_layer(Bottleneck, 256, res_layers[2], stride=2)
            layer4 = self._make_layer(Bottleneck, 512, res_layers[3], stride=1)
            conv5_pre = nn.Conv2d(2048, 1024, kernel_size=1, bias=False)
            conv5 = nn.Conv2d(1024, c7_channel, kernel_size=3, padding=1, bias=False)
            self.conv4_3_layer = 6

        bn5_pre = nn.BatchNorm2d(conv5_pre.out_channels)
        # conv5 = nn.Conv2d(1024, c7_channel, kernel_size=3, stride=2, padding=1, bias=False)
        bn5 = nn.BatchNorm2d(conv5.out_channels)
        self.backbone = nn.ModuleList([conv1, bn1, relu, maxpool, layer1, layer2, layer3,
                                     layer4, avgpool, conv5_pre, bn5_pre,relu, conv5, bn5, relu])
        print(self.backbone )
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
        if backbone == 'ResNet18':
            out_channels = [layer4[-1].conv2.out_channels,]
        else:
            out_channels = [layer2[-1].conv3.out_channels,]
        out_channels += [conv5.out_channels,
                        extra_layers[1][0].out_channels, extra_layers[3][0].out_channels,
                        extra_layers[5][0].out_channels, extra_layers[7][0].out_channels]
        if self.size == 512:
            out_channels.append(extra_layers[9][0].out_channels)

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
            self.softmax = nn.Softmax()
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

        sources = list()
        loc = list()
        conf = list()

        for i in range(self.conv4_3_layer):
            x = self.backbone[i](x)
        s = self.L2Norm(x)
        sources.append(s)
        for i in range(self.conv4_3_layer, len(self.backbone)):
            x = self.backbone[i](x)
        sources.append(x)

        for k, v in enumerate(self.extras):
            x = v(x)
            if k % 2 == 1:
                sources.append(x)

        for (f, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(f).permute(0, 2, 3, 1).contiguous()) # [ith_multi_layer, batch, height, width, out_channel]
            conf.append(c(f).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                self.priors.type(type(x.data))
                # self.priors,                 # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors.type(type(x.data))
            )
        return output, None

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

# extras = {
#     '300': [256, 'S', 512, 256, 'S', 512, 256, 512, 256, 512],
#     '512': [256, 'S', 512, 256, 'S', 512, 256, 'S', 512, 256, 'S', 512],
# }

class TSSD_ResNet(nn.Module):

    def __init__(self, phase, backbone, res_layers, extra_cfg, mb_cfg, num_classes, size=300, attention=False, c7_channel=1024, top_k=200, thresh=0.01, nms_thresh=0.45):
        super(TSSD_ResNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size
        self.attention_flag = attention
        self.priorbox = PriorBox(v2)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        # ResNet
        self.inplanes = 64
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU(inplace=True)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        avgpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        if backbone == 'ResNet18':
            layer1 = self._make_layer(BasicBlock, 64, res_layers[0])
            layer2 = self._make_layer(BasicBlock, 128, res_layers[1], stride=1)
            layer3 = self._make_layer(BasicBlock, 256, res_layers[2], stride=1)
            layer4 = self._make_layer(BasicBlock, 512, res_layers[3], stride=2)
            conv5_pre = nn.Conv2d(512, 1024, kernel_size=1, bias=False)
            conv5 = nn.Conv2d(1024, c7_channel, kernel_size=3, stride=2, padding=1, bias=False)
            self.conv4_3_layer = 8
        else:
            layer1 = self._make_layer(Bottleneck, 64, res_layers[0])
            layer2 = self._make_layer(Bottleneck, 128, res_layers[1], stride=2)
            layer3 = self._make_layer(Bottleneck, 256, res_layers[2], stride=2)
            layer4 = self._make_layer(Bottleneck, 512, res_layers[3], stride=1)
            conv5_pre = nn.Conv2d(2048, 1024, kernel_size=1, bias=False)
            conv5 = nn.Conv2d(1024, c7_channel, kernel_size=3, padding=1, bias=False)
            self.conv4_3_layer = 6
        bn5_pre = nn.BatchNorm2d(conv5_pre.out_channels)
        bn5 = nn.BatchNorm2d(conv5.out_channels)
        self.backbone = nn.ModuleList([conv1, bn1, relu, maxpool, layer1, layer2, layer3,
                                       layer4, avgpool, conv5_pre, bn5_pre, relu, conv5, bn5, relu])
        # extra
        extra_layers = []
        flag = False
        in_channels = conv5.out_channels
        for k, v in enumerate(extra_cfg):
            if in_channels != 'S':
                if v == 'S':
                    extra_layers.append(nn.Sequential(nn.Conv2d(in_channels, extra_cfg[k + 1],
                                                                kernel_size=(1, 3)[flag], stride=2, padding=1,
                                                                bias=False),
                                                      nn.BatchNorm2d(extra_cfg[k + 1]), nn.ReLU(inplace=True)))
                else:
                    extra_layers.append(
                        nn.Sequential(nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag], bias=False),
                                      nn.BatchNorm2d(v),
                                      nn.ReLU(inplace=True)))
                flag = not flag
            in_channels = v
        if self.size == 512:
            extra_layers.append(nn.Sequential(nn.Conv2d(in_channels, int(extra_cfg[-1] / 2), kernel_size=1, stride=1),
                                              nn.BatchNorm2d(int(extra_cfg[-1] / 2)),
                                              nn.ReLU(inplace=True)))
            extra_layers.append(
                nn.Sequential(nn.Conv2d(int(extra_cfg[-1] / 2), extra_cfg[-1], kernel_size=4, stride=1, padding=1),
                              nn.BatchNorm2d(extra_cfg[-1]),
                              nn.ReLU(inplace=True)))
        self.extras = nn.ModuleList(extra_layers)
        # multi_box
        if backbone == 'ResNet18':
            out_channels = [layer4[-1].conv2.out_channels,]
        else:
            out_channels = [layer2[-1].conv3.out_channels,]
        out_channels += [conv5.out_channels,
                        extra_layers[1][0].out_channels, extra_layers[3][0].out_channels,
                        extra_layers[5][0].out_channels, extra_layers[7][0].out_channels]
        if self.size == 512:
            out_channels.append(extra_layers[9][0].out_channels)
        loc_layers = []
        conf_layers = []

        for o, v in zip(out_channels, mb_cfg):
            loc_layers += [nn.Conv2d(o, v * 4, kernel_size=3, padding=1, bias=False)]
            conf_layers += [nn.Conv2d(o, v * num_classes, kernel_size=3, padding=1, bias=False)]
        self.loc = nn.ModuleList(loc_layers)
        self.conf = nn.ModuleList(conf_layers)

        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)

        # RNN
        self.rnn = nn.ModuleList([ConvLSTMCell(512,512,phase=phase), ConvLSTMCell(256,256,phase=phase)])
        print(self.rnn)
        if self.attention_flag:
            in_channel = 512
            self.attention = nn.ModuleList([ConvAttention(in_channel*2), ConvAttention(in_channel)])
            print(self.attention)
        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 0, top_k=top_k, conf_thresh=thresh, nms_thresh=nms_thresh)

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

    def forward(self, tx, state=None, init_tub=False):
        if self.phase == "train":
            rnn_state = [None] * 6
            seq_output = list()
            # seq_sources = list()
            seq_a_map = []
            for time_step in range(tx.size(1)):
                x = tx[:,time_step]
                sources = list()
                loc = list()
                conf = list()
                a_map = list()

                # apply vgg up to conv4_3 relu
                for i in range(self.conv4_3_layer):
                    x = self.backbone[i](x)
                s = self.L2Norm(x)
                sources.append(s)
                for i in range(self.conv4_3_layer, len(self.backbone)):
                    x = self.backbone[i](x)
                sources.append(x)

                for k, v in enumerate(self.extras):
                    x = v(x)
                    if k % 2 == 1:
                        sources.append(x)

                # seq_sources.append(sources)
                # apply multibox head to source layers
                if self.attention_flag:
                   for i, (x, l, c) in enumerate(zip(sources, self.loc, self.conf)):
                        if time_step == 0:
                            rnn_state[i] = self.rnn[i // 3].init_state(x)
                        a_map.append(self.attention[i//3](torch.cat((x, rnn_state[i][-1]),1)))
                        # a_map.append(a(x))
                        a_feat =  x *a_map[-1]
                        # a_feat = self.o_ratio*x + self.a_ratio*x*(a_map[-1])
                        rnn_state[i] = self.rnn[i//3](a_feat, rnn_state[i])
                        conf.append(c(rnn_state[i][-1]).permute(0, 2, 3, 1).contiguous())
                        loc.append(l(rnn_state[i][-1]).permute(0, 2, 3, 1).contiguous())
                else:
                    for i, (x, l, c) in enumerate(zip(sources, self.loc, self.conf)):
                        rnn_state[i] = self.rnn[i//3](x, rnn_state[i])
                        conf.append(c(rnn_state[i][-1]).permute(0, 2, 3, 1).contiguous())
                        loc.append(l(rnn_state[i][-1]).permute(0, 2, 3, 1).contiguous())

                loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
                conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

                output = (
                    loc.view(loc.size(0), -1, 4),
                    conf.view(conf.size(0), -1, self.num_classes),
                    self.priors.type(type(x.data))
                )
                seq_output.append(output)
                seq_a_map.append(tuple(a_map))

            return tuple(seq_output), tuple(seq_a_map)
        elif self.phase == 'test':

            sources = list()
            loc = list()
            conf = list()
            a_map = list()
            # apply vgg up to conv4_3 relu
            for i in range(6):
                tx = self.backbone[i](tx)
            s = self.L2Norm(tx)
            sources.append(s)
            for i in range(6, len(self.backbone)):
                tx = self.backbone[i](tx)
            sources.append(tx)

            for k, v in enumerate(self.extras):
                tx = v(tx)
                if k % 2 == 1:
                    sources.append(tx)

            # apply multibox head to source layers
            if self.attention_flag:
                for i, (x, l, c) in enumerate(zip(sources, self.loc, self.conf)):
                    if state[i] is None:
                        state[i] = self.rnn[i // 3].init_state(x)
                    a_map.append(self.attention[i // 3](torch.cat((x, state[i][-1]), 1)))
                    a_feat = x * a_map[-1]
                    state[i] = self.rnn[i // 3](a_feat, state[i])
                    conf.append(c(state[i][-1]).permute(0, 2, 3, 1).contiguous())
                    loc.append(l(state[i][-1]).permute(0, 2, 3, 1).contiguous())
            else:
                for i, (x, l, c) in enumerate(zip(sources, self.loc, self.conf)):
                    state[i] = self.rnn[i//3](x, state[i])
                    conf.append(c(state[i][-1]).permute(0, 2, 3, 1).contiguous())
                    loc.append(l(state[i][-1]).permute(0, 2, 3, 1).contiguous())

            loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
            conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
            # if self.tub:
            #     for a_idx, a in enumerate(a_map[:3]):
            #         if not a_idx:
            #             tub_tensor = a
            #             tub_tensor_size = a.size()[2:]
            #         else:
            #             tub_tensor = torch.cat((tub_tensor, F.upsample(a, tub_tensor_size, mode='bilinear')), dim=1)
            #     if init_tub:
            #         self.detect.init_tubelets()
            #     output = self.detect(
            #         loc.view(loc.size(0), -1, 4),  # loc preds
            #         self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            #         self.priors.type(type(tx.data)),  # default boxes
            #         tub_tensor
            #     )
            # else:
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                self.priors.type(type(tx.data))
            )

            return output, state, tuple(a_map)


    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

# mbox = {
#     'VOC_300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
#     'MOT_300': [5, 5, 5, 5, 5, 5],
#     'VOC_512': [6, 6, 6, 6, 6, 4, 4],
# }
#
#
# extras = {
#     '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
#     '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
# }

def build_net(phase, backbone='ResNet18',prior='VOC_300', size=300, num_classes=21, pm=0., c7_channel=1024, tssd=False, attention=False,
              top_k=200, thresh=0.01, nms_thresh=0.45):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size not in [300, 512]:
        print("Error: Sorry only SSD300/512 is supported currently!")
        return
    if backbone == 'ResNet18':
        res_layers = [2,2,2,2]
    elif backbone == 'ResNet50':
        res_layers = [3, 4, 6, 3]
    elif backbone == 'ResNet101':
        res_layers = [3, 4, 23, 3]
    else:
        print("Error: Unknown model!")
        return
    if tssd:
        return TSSD_ResNet(phase, backbone, res_layers, extras[str(size)], mbox[prior], num_classes, size=size, c7_channel=c7_channel, attention=attention, top_k=top_k, thresh=thresh, nms_thresh=nms_thresh)
    else:
        return SSD_ResNet(phase, backbone, res_layers, extras[str(size)], mbox[prior], num_classes,size=size, c7_channel=c7_channel, top_k=top_k, thresh=thresh, nms_thresh=nms_thresh)

