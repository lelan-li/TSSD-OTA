import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import v2
import os

class SSD(nn.Module):
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

    def __init__(self, phase, base, extras, head, num_classes, top_k=200, thresh=0.01, nms_thresh=0.45):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.priorbox = PriorBox(v2)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = 300

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 0, top_k=top_k, conf_thresh=thresh, nms_thresh=nms_thresh)
                                # num_classes, bkg_label, top_k, conf_thresh, nms_thresh

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

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous()) # [ith_multi_layer, batch, height, width, out_channel]
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

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
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size), volatile=(False, True)[self.phase=='test']),
                Variable(torch.zeros(state_size), volatile=(False, True)[self.phase=='test'])
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
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

        return hidden, cell

class TSSD(nn.Module):

    def __init__(self, phase, base, extras, head, num_classes, lstm='lstm', top_k=200,thresh= 0.01,nms_thresh=0.45):
        super(TSSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.priorbox = PriorBox(v2)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = 300

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.lstm_mode = lstm

        self.reduce = nn.ModuleList(head[0][0::2])
        self.lstm = nn.ModuleList(head[1][0::2])
        self.loc = nn.ModuleList(head[0][1::2])
        self.conf = nn.ModuleList(head[1][1::2])

        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 0, top_k=top_k, conf_thresh=thresh, nms_thresh=nms_thresh)
            # num_classes, bkg_label, top_k, conf_thresh, nms_thresh

    def forward(self, tx, lstm_state=None):
        if self.phase == "train":
            lstm_state = [None] * len(self.lstm)
            seq_output = []
            for time_step in range(tx.size(1)):
                x = tx[:,time_step,:,:]
                sources = list()
                loc = list()
                conf = list()

                # apply vgg up to conv4_3 relu
                for k in range(23):
                    x = self.vgg[k](x)

                s = self.L2Norm(x)
                sources.append(s)

                # apply vgg up to fc7
                for k in range(23, len(self.vgg)):
                    x = self.vgg[k](x)
                sources.append(x)

                # apply extra layers and cache source layer outputs
                for k, v in enumerate(self.extras):
                    x = F.relu(v(x), inplace=True)
                    if k % 2 == 1:
                        sources.append(x)

                # apply multibox head to source layers
                for i, (x, r, lstm, l, c) in enumerate(zip(sources, self.reduce, self.lstm, self.loc, self.conf)):
                #     if time_step:
                #         if i==0:
                #             post_scale = torch.cuda.FloatTensor(lstm_state[i+1][0].data.size())
                #             lstm_state[i+1][0].data.copy_(post_scale)
                #             post_scale.resize_as_(lstm_state[i][0].data)
                #             lstm_state[i][0].data = lstm_state[i][0].data *0.75 + post_scale*0.25
                #         elif i==5:
                #             pre_scale = torch.cuda.FloatTensor(lstm_state[i - 1][0].data.size())
                #             lstm_state[i - 1][0].data.copy_(pre_scale)
                #             pre_scale.resize_as_(lstm_state[i][0].data)
                #             lstm_state[i][0].data  = lstm_state[i][0].data *0.75 + pre_scale*0.25
                #         else:
                #             post_scale = torch.cuda.FloatTensor(lstm_state[i + 1][0].data.size())
                #             pre_scale = torch.cuda.FloatTensor(lstm_state[i - 1][0].data.size())
                #             lstm_state[i + 1][0].data.copy_(post_scale)
                #             lstm_state[i - 1][0].data.copy_(pre_scale)
                #             post_scale.resize_as_(lstm_state[i][0].data)
                #             pre_scale.resize_as_(lstm_state[i][0].data)
                #             lstm_state[i][0].data  = lstm_state[i][0].data *0.5 + post_scale*0.25 \
                #                                + pre_scale*0.25
                    lstm_state[i] = lstm(r(x), lstm_state[i])
                    loc.append(l(lstm_state[i][0]).permute(0, 2, 3, 1).contiguous())
                    conf.append(c(lstm_state[i][0]).permute(0, 2, 3, 1).contiguous())

                loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
                conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
                output = (
                    loc.view(loc.size(0), -1, 4),
                    conf.view(conf.size(0), -1, self.num_classes),
                    self.priors
                )
                seq_output.append(output)
            return tuple(seq_output)
        elif self.phase == 'test':

            # lstm_state  = state

            sources = list()
            loc = list()
            conf = list()

            # apply vgg up to conv4_3 relu
            for k in range(23):
                tx = self.vgg[k](tx)

            s = self.L2Norm(tx)
            sources.append(s)

            # apply vgg up to fc7
            for k in range(23, len(self.vgg)):
                tx = self.vgg[k](tx)
            sources.append(tx)

            # apply extra layers and cache source layer outputs
            for k, v in enumerate(self.extras):
                tx = F.relu(v(tx), inplace=True)
                if k % 2 == 1:
                    sources.append(tx)

            # apply multibox head to source layers
            for i, (x, r, lstm, l, c) in enumerate(zip(sources, self.reduce, self.lstm, self.loc, self.conf)):
                lstm_state[i] = lstm(r(x), lstm_state[i])
                loc.append(l(lstm_state[i][0]).permute(0, 2, 3, 1).contiguous())
                conf.append(c(lstm_state[i][0]).permute(0, 2, 3, 1).contiguous())

            loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
            conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                self.priors.type(type(tx.data))  # default boxes
            )
            return output, lstm_state


    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
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
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def multibox(vgg, extra_layers, cfg, num_classes, lstm=None, phase='train'):
    loc_layers = []
    conf_layers = []
    vgg_source = [24, -2]
    for k, v in enumerate(vgg_source):

        if lstm in ['lstm']:
            loc_layers += [ nn.Conv2d(vgg[v].out_channels, int(vgg[v].out_channels/2), kernel_size=1),
                            nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
            # loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [ConvLSTMCell(int(vgg[v].out_channels/2), vgg[v].out_channels, phase=phase),
                            nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                     cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels,
                            cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):

        if lstm in ['lstm']:
            loc_layers += [nn.Conv2d(v.out_channels, int(v.out_channels/2), kernel_size=1),
                           nn.Conv2d(v.out_channels, cfg[k]* 4, kernel_size=3, padding=1)]
            # loc_layers += [nn.Conv2d(v.out_channels, cfg[k]* 4, kernel_size=3, padding=1)]
            conf_layers += [ConvLSTMCell(int(v.out_channels/2), v.out_channels, phase=phase),
                            nn.Conv2d(v.out_channels, cfg[k]* num_classes, kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                      * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512], # output channel
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21, tssd='ssd', top_k=200, thresh=0.01, nms_thresh=0.45):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300:
        print("Error: Sorry only SSD300 is supported currently!")
        return
    if tssd == 'ssd':
        return SSD(phase, *multibox(vgg(base[str(size)], 3),
                                    add_extras(extras[str(size)], 1024),
                                    mbox[str(size)], num_classes, phase=phase), num_classes,
                   top_k=top_k,thresh= thresh,nms_thresh=nms_thresh)
    else:
        return TSSD(phase, *multibox(vgg(base[str(size)], 3),
                                add_extras(extras[str(size)], 1024),
                                mbox[str(size)], num_classes, lstm=tssd, phase=phase), num_classes, lstm=tssd,
                    top_k=top_k, thresh=thresh, nms_thresh=nms_thresh)
