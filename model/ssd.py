import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from data import VOC_300, MOT_300, VOC_512
import os
from .networks import vgg, add_extras, ConvAttention, ConvLSTMCell, ConvGRUCell

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

    def __init__(self, phase, base, extras, head, num_classes, size=300, top_k=200, thresh=0.01, nms_thresh=0.45, attention=False, prior='v2', device=torch.device('cuda')):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.attention_flag = attention
        self.device = device
        # TODO: implement __call__ in PriorBox
        if prior=='VOC_300':
            self.priorbox = PriorBox(VOC_300)
        elif prior=='MOT_300':
            self.priorbox = PriorBox(MOT_300)
        elif prior=='VOC_512':
            self.priorbox = PriorBox(VOC_512)

        with torch.no_grad():
            self.priors = self.priorbox.forward().to(self.device)
        self.size = size

        # SSD network
        self.backbone = nn.ModuleList(base)
        self.conv4_3_layer = (23, 33)[len(self.backbone)>40]
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.extras_skip = (2, 3)[len(self.backbone)>40]
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.attention_flag:
            self.attention = nn.ModuleList([ConvAttention(512),ConvAttention(256)])
                                            # ConvAttention(512),ConvAttention(256),
                                            # ConvAttention(256),ConvAttention(256)])
            print(self.attention)
        if phase == 'test':
            self.softmax = nn.Softmax(dim=1)
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
        a_map = list()
        # apply vgg up to conv4_3 relu
        for k in range(self.conv4_3_layer):
            x = self.backbone[k](x)
        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(self.conv4_3_layer, len(self.backbone)):
            x = self.backbone[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % self.extras_skip == 1:
                sources.append(x)

        # apply multibox head to source layers
        if self.attention_flag:
            for i, (f, l, c) in enumerate(zip(sources, self.loc, self.conf)):
                a_map.append(self.attention[i//3](f))
                a_feat = f*a_map[-1]
                loc.append(l(a_feat).permute(0, 2, 3, 1).contiguous())  # [ith_multi_layer, batch, height, width, out_channel]
                conf.append(c(a_feat).permute(0, 2, 3, 1).contiguous())
        else:
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

class TSSD(nn.Module):

    def __init__(self, phase, base, extras, head, num_classes, lstm='lstm', size=300,
                 top_k=200,thresh= 0.01,nms_thresh=0.45, attention=False, prior='v2', cuda=True,
                 tub=0, tub_thresh=1.0, tub_generate_score=0.7, device=torch.device('cpu')):
        super(TSSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.device = device

        # TODO: implement __call__ in PriorBox
        if prior=='VOC_300':
            self.priorbox = PriorBox(VOC_300)
        elif prior=='MOT_300':
            self.priorbox = PriorBox(MOT_300)
        elif prior=='VOC_512':
            self.priorbox = PriorBox(VOC_512)

        with torch.no_grad():
            self.priors = self.priorbox.forward().to(self.device)
        self.size = size
        self.attention_flag = attention
        # SSD network
        self.backbone = nn.ModuleList(base)
        self.conv4_3_layer = (23, 33)[len(self.backbone)>40]
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.extras_skip = (2, 3)[len(self.backbone)>40]
        self.lstm_mode = lstm

        self.rnn = nn.ModuleList(head[2])
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        print(self.rnn)
        if self.attention_flag:
            in_channel = 512
            self.attention = nn.ModuleList([ConvAttention(in_channel*2), ConvAttention(in_channel)])
            print(self.attention)
        if phase == 'test':
            self.tub = tub
            self.softmax = nn.Softmax(dim=1)
            self.detect = Detect(num_classes, 0, top_k=top_k, conf_thresh=thresh, nms_thresh=nms_thresh,
                                 tub=tub, tub_thresh=tub_thresh, tub_generate_score=tub_generate_score)

    def forward(self, tx, state=None, init_tub=False):
        if self.phase == "train":
            rnn_state = [None] * 6
            seq_output = list()
            seq_sources = list()
            seq_a_map = []
            for time_step in range(tx.size(1)):
                x = tx[:,time_step]
                sources = list()
                loc = list()
                conf = list()
                a_map = list()

                # apply vgg up to conv4_3 relu
                for k in range(self.conv4_3_layer):
                    x = self.backbone[k](x)

                s = self.L2Norm(x)
                sources.append(s)

                # apply vgg up to fc7
                for k in range(23, len(self.backbone)):
                    x = self.backbone[k](x)
                sources.append(x)

                # apply extra layers and cache source layer outputs
                for k, v in enumerate(self.extras):
                    x = F.relu(v(x), inplace=True)
                    if k % self.extras_skip == 1:
                        sources.append(x)
                seq_sources.append(sources)
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
                        c_current = c(rnn_state[i][-1])
                        conf.append(c(rnn_state[i][-1]).permute(0, 2, 3, 1).contiguous())
                        loc.append(l(rnn_state[i][-1]).permute(0, 2, 3, 1).contiguous())
                        c_previous = c_current
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
                    self.priors,
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
            for k in range(self.conv4_3_layer):
                tx = self.backbone[k](tx)

            s = self.L2Norm(tx)
            sources.append(s)

            # apply vgg up to fc7
            for k in range(23, len(self.backbone)):
                tx = self.backbone[k](tx)
            sources.append(tx)

            # apply extra layers and cache source layer outputs
            for k, v in enumerate(self.extras):
                tx = F.relu(v(tx), inplace=True)
                if k % self.extras_skip == 1:
                    sources.append(tx)

            # apply multibox head to source layers
            if self.attention_flag:
                for i, (x, l, c) in enumerate(zip(sources, self.loc, self.conf)):
                    if state[i] is None:
                        state[i] = self.rnn[i // 3].init_state(x)
                    a_map.append(self.attention[i // 3](torch.cat((x, state[i][-1]), 1)))
                    # a_map.append(a(x))
                    # a_feat = self.o_ratio * x + self.a_ratio * x * (a_map[-1])
                    a_feat = x * a_map[-1]
                    state[i] = self.rnn[i // 3](a_feat, state[i])
                    # conf.append(c(a_feat).permute(0, 2, 3, 1).contiguous())
                    # loc.append(l(a_feat).permute(0, 2, 3, 1).contiguous())
                    conf.append(c(state[i][-1]).permute(0, 2, 3, 1).contiguous())
                    loc.append(l(state[i][-1]).permute(0, 2, 3, 1).contiguous())
            else:
                for i, (x, l, c) in enumerate(zip(sources, self.loc, self.conf)):
                    state[i] = self.rnn[i//3](x, state[i])
                    conf.append(c(state[i][-1]).permute(0, 2, 3, 1).contiguous())
                    loc.append(l(state[i][-1]).permute(0, 2, 3, 1).contiguous())

            loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
            conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
            if self.tub:
                with torch.no_grad():
                    for a_idx, a in enumerate(a_map[:3]):
                        if not a_idx:
                            tub_tensor = a
                            tub_tensor_size = a.size()[2:]
                        else:
                            tub_tensor = torch.cat((tub_tensor, F.upsample(a, tub_tensor_size, mode='bilinear', align_corners=True)), dim=1)
                if init_tub:
                    self.detect.init_tubelets()
                output = self.detect(
                    loc.view(loc.size(0), -1, 4),  # loc preds
                    self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                    self.priors,  # default boxes
                    tub_tensor
                )
            else:
                output = self.detect(
                    loc.view(loc.size(0), -1, 4),  # loc preds
                    self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                    self.priors,  # default boxes
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

def multibox(vgg, extra_layers, cfg, num_classes, lstm=None, phase='train', batch_norm=False):
    loc_layers = []
    conf_layers = []
    rnn_layer = []
    vgg_source = ([24, -2], [34, -3])[batch_norm==True]
    for k, v in enumerate(vgg_source):

        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    key_extra_layers = (extra_layers[1::2], extra_layers[1::3])[batch_norm==True]
    for k, v in enumerate(key_extra_layers, 2):

        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    if lstm in ['tblstm']:
        rnn_layer = [ConvLSTMCell(512,512,phase=phase), ConvLSTMCell(256,256,phase=phase)]
    elif lstm in ['gru']:
        rnn_layer = [ConvGRUCell(512,512,phase=phase), ConvGRUCell(256,256,phase=phase)]
    return vgg, extra_layers, (loc_layers, conf_layers, rnn_layer)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512], # output channel
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


def build_ssd(phase, size=300, num_classes=21, tssd='ssd', top_k=200, thresh=0.01, prior='VOC_300', bn=False,
              nms_thresh=0.45, attention=False, tub=0, tub_thresh=1.0, tub_generate_score=0.7, device=torch.device('cpu')):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size not in [300, 512]:
        print("Error: Sorry only SSD300 is supported currently!")
        return
    if tssd == 'ssd':
        return SSD(phase, *multibox(vgg(base[str(size)], 3, batch_norm=bn),
                                    add_extras(extras[str(size)], 512, batch_norm=bn, size=size),
                                    mbox[prior], num_classes, phase=phase, batch_norm=bn), num_classes,size=size,
                   top_k=top_k,thresh= thresh,nms_thresh=nms_thresh, attention=attention, prior=prior, device=device)
    else:
        return TSSD(phase, *multibox(vgg(base[str(size)], 3, batch_norm=bn),
                                add_extras(extras[str(size)], 512, size=size),
                                mbox[prior], num_classes, lstm=tssd, phase=phase,batch_norm=bn),
                    num_classes, lstm=tssd, size=size, top_k=top_k, thresh=thresh, prior=prior,
                    nms_thresh=nms_thresh, attention=attention,
                    tub=tub, tub_thresh=tub_thresh, tub_generate_score=tub_generate_score, device=device
                    )
