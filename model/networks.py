import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import cv2

vgg_base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512], # output channel
    '320': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}

extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}
mbox = {
    'VOC_300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    'MOT_300': [5, 5, 5, 5, 5, 5],
    'VOC_512': [6, 6, 6, 6, 6, 4, 4],
}

def xavier(param):
    init.xavier_uniform_(param)

def orthogonal(param):
    init.orthogonal_(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def conv_weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def orthogonal_weights_init(m):
    if isinstance(m, nn.Conv2d):
        orthogonal(m.weight.data)
        m.bias.data.fill_(1)

def net_init(ssd_net, backbone,resume_from_ssd='ssd', tssd='ssd', attention=False, pm=0.0, refine=False):
    if resume_from_ssd != 'ssd':
        if attention:
            print('Initializing Attention weights...')
            ssd_net.attention.apply(conv_weights_init)
        if tssd in ['gru', 'tblstm']:
            print('Initializing RNN weights...')
            ssd_net.rnn.apply(orthogonal_weights_init)
    else:
        print('Initializing extra, loc, conf weights...')
        # initialize newly added layers' weights with xavier method
        if backbone in ['RFB_VGG'] or backbone[:6] == 'ResNet':
            ssd_net.extras.apply(conv_weights_init)
            ssd_net.loc.apply(conv_weights_init)
            ssd_net.conf.apply(conv_weights_init)
            if pm != 0.0:
                ssd_net.pm.apply(conv_weights_init)
        elif backbone in ['RefineDet_VGG']:
            ssd_net.extras.apply(weights_init)
            ssd_net.trans_layers.apply(weights_init)
            ssd_net.latent_layrs.apply(weights_init)
            ssd_net.up_layers.apply(weights_init)
            ssd_net.odm_loc.apply(weights_init)
            ssd_net.odm_conf.apply(weights_init)
            if refine:
                ssd_net.arm_loc.apply(weights_init)
                ssd_net.arm_conf.apply(weights_init)
            if pm != 0.0:
                ssd_net.pm.apply(conv_weights_init)
        else:
            ssd_net.extras.apply(weights_init)
            ssd_net.loc.apply(weights_init)
            ssd_net.conf.apply(weights_init)
        if tssd in ['gru', 'tblstm']:
            print('Initializing RNN weights...')
            ssd_net.rnn.apply(orthogonal_weights_init)
        if attention:
            print('Initializing Attention weights...')
            ssd_net.attention.apply(conv_weights_init)
# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False, pool5_ds=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    if pool5_ds:
        pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    else:
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    # conv7 = nn.Conv2d(1024, 512, kernel_size=1)
    if batch_norm:
        layers += [pool5, conv6, nn.BatchNorm2d(conv6.out_channels),
                   nn.ReLU(inplace=True), conv7, nn.BatchNorm2d(conv7.out_channels), nn.ReLU(inplace=True)]
    else:
        layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False, size=300):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                if batch_norm:
                    layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1), nn.BatchNorm2d(cfg[k + 1])]
                else:
                    layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                         kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                if batch_norm and k in [7]:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag]), nn.BatchNorm2d(v)]

                else:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if size == 512:
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    return layers

class ConvAttention(nn.Module):

    def __init__(self, inchannel):
        super(ConvAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            # nn.ConvTranspose2d(int(inchannel/2), int(inchannel/4), kernel_size=3, stride=2, padding=1, output_padding=0, bias=False),
            nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(inchannel, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, feats):
        return self.attention(feats)



# https://www.jianshu.com/p/72124b007f7d
class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, phase='train'):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, 3, padding=1)
        self.phase = phase

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.size()[0]
        spatial_size = input_.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.zeros(state_size, requires_grad=(True, False)[self.phase == 'test']).cuda(),
                torch.zeros(state_size, requires_grad=(True, False)[self.phase == 'test']).cuda(),
            )

        prev_cell, prev_hidden = prev_state
        # prev_hidden_drop = F.dropout(prev_hidden, training=(False, True)[self.phase=='train'])
        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((F.dropout(input_, p=0.2, training=(False,True)[self.phase=='train']), prev_hidden), 1)
        # stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = F.sigmoid(in_gate)
        remember_gate = F.sigmoid(remember_gate)
        out_gate = F.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = F.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * F.tanh(cell)

        return cell, hidden

    def init_state(self, input_):
        batch_size = input_.size()[0]
        spatial_size = input_.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)
        state = (
            torch.zeros(state_size, requires_grad=(True, False)[self.phase == 'test']).cuda(),
            torch.zeros(state_size, requires_grad=(True, False)[self.phase == 'test']).cuda(),
        )
        return state

class ConvJANET(nn.Module):
    """
    Generate a convolutional JANET cell
    """

    def __init__(self, input_size, hidden_size, phase='train'):
        super(ConvJANET, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 2 * hidden_size, 3, padding=1)
        self.phase = phase

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.size()[0]
        spatial_size = input_.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (torch.zeros(state_size, requires_grad=(True, False)[self.phase == 'test']).cuda(),)

        prev_cell = prev_state[-1]
        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((F.dropout(input_, p=0.2, training=(False,True)[self.phase=='train']), prev_cell), 1)
        # stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        remember_gate, cell_gate = gates.chunk(2, 1)

        # apply sigmoid non linearity
        remember_gate = F.sigmoid(remember_gate)
        # apply tanh non linearity
        cell_gate = F.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + ((1-remember_gate) * cell_gate)

        return (cell, )

    def init_state(self, input_):
        batch_size = input_.size()[0]
        spatial_size = input_.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)
        state = (torch.zeros(state_size, requires_grad=(True, False)[self.phase == 'test']).cuda(),)
        return state

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, cuda_flag=True, phase='train'):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.ConvGates = nn.Conv2d(self.input_size + self.hidden_size, 2 * self.hidden_size, 3,
                                   padding=self.kernel_size // 2)
        self.Conv_ct = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, 3, padding=self.kernel_size // 2)
        self.phase = phase

    def forward(self, input, pre_state):
        if pre_state is None:
            size_h = [input.size()[0], self.hidden_size] + list(input.size()[2:])
            pre_state = (torch.zeros(size_h, requires_grad=(True, False)[self.phase == 'test']).cuda(),)

        hidden = pre_state[-1]
        c1 = self.ConvGates(torch.cat((F.dropout(input,p=0.2,training=(False,True)[self.phase=='train']), hidden), 1))
        (rt, ut) = c1.chunk(2, 1)
        reset_gate = F.sigmoid(rt)
        update_gate = F.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate, hidden)
        p1 = self.Conv_ct(torch.cat((input, gated_hidden), 1))
        ct = F.tanh(p1)
        next_h = torch.mul(update_gate, hidden) + (1 - update_gate) * ct
        return (next_h, )

    def init_state(self, input):
        size_h = [input.size()[0], self.hidden_size] + list(input.size()[2:])
        state = torch.zeros(size_h, requires_grad=(True, False)[self.phase == 'test']).cuda(),
        return state

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
#
class ResNetSSD(nn.Module):

    def __init__(self, block, layers, extra_cfg, mb_cfg, num_classes):
        self.inplanes = 64
        super(ResNetSSD, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu =  nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv5_pre = nn.Conv2d(2048, 1024, kernel_size=1, bias=False)
        self.bn5_pre = nn.BatchNorm2d(self.conv5_pre.out_channels)
        self.conv5 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1)
        self.bn5 = nn.BatchNorm2d(2048)
        self.size = 512
        # extra
        extra_layers = []
        flag = False
        in_channels = self.conv5.out_channels
        for k, v in enumerate(extra_cfg):
            if in_channels != 'S':
                if v == 'S':
                    extra_layers += [nn.Sequential(nn.Conv2d(in_channels, extra_cfg[k + 1],
                                             kernel_size=(1, 3)[flag], stride=2, padding=1), nn.BatchNorm2d(extra_cfg[k + 1]), nn.ReLU(inplace=True))]
                else:
                    extra_layers += [nn.Sequential(nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag]), nn.BatchNorm2d(v), nn.ReLU(inplace=True))]
                flag = not flag
            in_channels = v
        if self.size == 512:
            extra_layers.append(nn.Sequential(nn.Conv2d(in_channels, int(extra_cfg[-1]/2), kernel_size=1, stride=1), nn.BatchNorm2d(int(extra_cfg[-1]/2)),
                                      nn.ReLU(inplace=True)))
            extra_layers.append(nn.Sequential(nn.Conv2d(int(extra_cfg[-1]/2), extra_cfg[-1], kernel_size=4, stride=1, padding=1), nn.BatchNorm2d(extra_cfg[-1]),
                                      nn.ReLU(inplace=True)))

        self.extra_layers = nn.ModuleList(extra_layers)
        print(self.extra_layers)

        # multi_box
        loc_layers = []
        conf_layers = []
        out_channels = [self.layer2[-1].conv3.out_channels, self.conv5.out_channels,
                        self.extra_layers[1][0].out_channels, self.extra_layers[3][0].out_channels,
                        self.extra_layers[5][0].out_channels, self.extra_layers[7][0].out_channels]
        if self.size == 512:
            out_channels.append(self.extra_layers[9][0].out_channels)
        for o, v in zip(out_channels, mb_cfg):
            loc_layers += [nn.Conv2d(o, v * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(o, v * num_classes, kernel_size=3, padding=1)]
        self.loc_layers = nn.ModuleList(loc_layers)
        self.conf_layers = nn.ModuleList(conf_layers)

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

## prediction module in DSSD
class PreModule(nn.Module):

    def __init__(self, inchannl, channel_increment_factor):
        super(PreModule, self).__init__()
        self.inchannel = inchannl
        self.pm = nn.Sequential(
            nn.Conv2d(inchannl, inchannl, kernel_size=1),
            nn.Conv2d(inchannl, inchannl, kernel_size=1),
            nn.Conv2d(inchannl, int(inchannl*channel_increment_factor), kernel_size=1)
        )
        self.extend = nn.Conv2d(inchannl, int(inchannl*channel_increment_factor), kernel_size=1)

    def forward(self, x):
        return self.extend(x) + self.pm(x)

# if __name__ == '__main__':
#     # # resnet101:[3,4,22,3], resnet50:[3,4,6,3], resnet18:[2,2,2,2]
#     img = cv2.resize(cv2.imread('../demo/comp/ILSVRC2015_val_00020001/3.jpg'), (512,512))
#     img_torch = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).type(torch.FloatTensor).repeat(2,1,1,1)
#     img_torch -= 128.
#     img_torch /= 255.
#     print(img_torch.size())
#     extra_cfg = [256, 'S', 512, 256, 'S', 512, 256, 'S', 512, 256, 'S', 512] #[256, 'S', 512, 256, 'S', 512, 256, 512, 256, 512]
#     mb_cfg = [6, 6, 6, 6, 6, 4, 4] # [4, 6, 6, 6, 4, 4]
#     model = ResNetSSD(Bottleneck, [2, 2, 2, 2], extra_cfg, mb_cfg, 3)
#     x = model.conv1(img_torch)
#     x = model.bn1(x)
#     x = model.maxpool(x)
#
#     x = model.layer1(x)
#     x = model.layer2(x)
#     x = model.layer3(x)
#     x = model.layer4(x)
#
#     x = model.avgpool(x)
#     x = model.conv5_pre(x)
#
#     x = model.conv5(x)
#
#     for i, ex in enumerate(model.extra_layers):
#         print(i)
#         x = ex(x)
#
#     pass