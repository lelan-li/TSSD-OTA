import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
from .networks import vgg, vgg_base, ConvAttention

class RefineSSD(nn.Module):

    def __init__(self, size, num_classes, use_refine=False, phase='train', dropout=1.0, residual=False, channel=False):
        super(RefineSSD, self).__init__()
        self.num_classes = num_classes
        self.size = size
        self.use_refine = use_refine
        self.phase = phase
        if dropout>0. and dropout<1.:
            self.dropout = True
            self.dropout_p = dropout
        else:
            self.dropout = False
        self.box_num = 3
        if channel:
            if size==320:
                from data.config import VOC_320
                spatial_size = VOC_320['feature_maps']
            elif size==512:
                from data.config import VOC_512_RefineDet
                spatial_size = VOC_512_RefineDet['feature_maps']
        else:
            spatial_size = [None] * 4

        # SSD network
        self.backbone = nn.ModuleList(vgg(vgg_base['320'], 3, pool5_ds=True))
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm_4_3 = L2Norm(512, 10)
        self.L2Norm_5_3 = L2Norm(512, 8)
        self.last_layer_trans = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                              nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.extras = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))

        if self.use_refine:
            self.arm_loc = nn.ModuleList([nn.Conv2d(512, self.box_num *4, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(512, self.box_num *4, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(1024,self.box_num *4, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(512, self.box_num *4, kernel_size=3, stride=1, padding=1),
                                          ])
            self.arm_att = nn.ModuleList([ConvAttention(512, residual=residual, channel=channel, spatial_size=spatial_size[0]), ConvAttention(512, residual=residual, channel=channel, spatial_size=spatial_size[1]),
                                          ConvAttention(1024, residual=residual, channel=channel,spatial_size=spatial_size[2]), ConvAttention(512, residual=residual, channel=channel, spatial_size=spatial_size[3])])
        self.odm_loc = nn.ModuleList([nn.Conv2d(256, self.box_num *4, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(256, self.box_num *4, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(256, self.box_num *4, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(256, self.box_num *4, kernel_size=3, stride=1, padding=1),
                                      ])
        self.odm_conf = nn.ModuleList([nn.Conv2d(256, self.box_num *self.num_classes, kernel_size=3, stride=1, padding=1),
                                       nn.Conv2d(256, self.box_num *self.num_classes, kernel_size=3, stride=1, padding=1),
                                       nn.Conv2d(256, self.box_num *self.num_classes, kernel_size=3, stride=1, padding=1),
                                       nn.Conv2d(256, self.box_num *self.num_classes, kernel_size=3, stride=1, padding=1),
                                       ])
        self.trans_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                                                         nn.ReLU(inplace=True),
                                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
                                           nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                                                         nn.ReLU(inplace=True),
                                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
                                           nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                                                         nn.ReLU(inplace=True),
                                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
                                           ])
        self.up_layers = nn.ModuleList([nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
                                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
                                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0), ])
        self.latent_layrs = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                           nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                           nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                           ])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=1)
            self.istraining=False
        else:
            self.istraining=True

    def forward(self, x):

        arm_sources = list()
        arm_loc_list = list()
        arm_att_list = list()
        obm_loc_list = list()
        obm_conf_list = list()
        obm_sources = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.backbone[k](x)
        if self.dropout:
            x = F.dropout(x, self.dropout_p, training=self.istraining)
        norm_s = self.L2Norm_4_3(x)
        x, att_map = self.arm_att[0](norm_s)
        arm_sources.append(x)
        arm_att_list.append(att_map)

        # apply vgg up to conv5_3
        for k in range(23, 30):
            x = self.backbone[k](x)
        if self.dropout:
            x = F.dropout(x, self.dropout_p, training=self.istraining)
        norm_s = self.L2Norm_5_3(x)
        x, att_map = self.arm_att[1](norm_s)
        arm_sources.append(x)
        arm_att_list.append(att_map)

        # apply vgg up to fc7
        for k in range(30, len(self.backbone)):
            x = self.backbone[k](x)
        if self.dropout:
            x = F.dropout(x, self.dropout_p, training=self.istraining)
        x, att_map = self.arm_att[2](x)
        arm_sources.append(x)
        arm_att_list.append(att_map)

        # conv6_2
        x = self.extras(x)
        if self.dropout:
            x = F.dropout(x, self.dropout_p, training=self.istraining)
        x, att_map = self.arm_att[3](x)
        arm_sources.append(x)
        arm_att_list.append(att_map)
        # apply multibox head to arm branch
        if self.use_refine:
            for (a, l) in zip(arm_sources, self.arm_loc):
                arm_loc_list.append(l(a).permute(0, 2, 3, 1).contiguous())
            arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc_list], 1)
        x = self.last_layer_trans(x)
        if self.dropout:
            x = F.dropout(x, self.dropout_p, training=self.istraining)
        obm_sources.append(x)

        # get transformed layers
        trans_layer_list = list()
        for (x_t, t) in zip(arm_sources, self.trans_layers):
            trans_layer_list.append(t(x_t))
        # fpn module
        trans_layer_list.reverse()
        arm_sources.reverse()
        for (t, u, l) in zip(trans_layer_list, self.up_layers, self.latent_layrs):
            x = F.relu(l(F.relu(u(x) + t, inplace=True)), inplace=True)
            if self.dropout:
                x = F.dropout(x, self.dropout_p, training=self.istraining)
            obm_sources.append(x)
        obm_sources.reverse()
        for (s, l, c) in zip(obm_sources, self.odm_loc, self.odm_conf):
            obm_loc_list.append(l(s).permute(0, 2, 3, 1).contiguous())
            obm_conf_list.append(c(s).permute(0, 2, 3, 1).contiguous())
        obm_loc = torch.cat([o.view(o.size(0), -1) for o in obm_loc_list], 1)
        obm_conf = torch.cat([o.view(o.size(0), -1) for o in obm_conf_list], 1)
        # apply multibox head to source layers

        if self.phase == 'test':
            if self.use_refine:
                output = (
                    arm_loc.view(arm_loc.size(0), -1, 4),  # loc preds
                    tuple(arm_att_list),  # conf preds
                    obm_loc.view(obm_loc.size(0), -1, 4),  # loc preds
                    self.softmax(obm_conf.view(-1, self.num_classes)),  # conf preds
                )
            else:
                output = (
                    obm_loc.view(obm_loc.size(0), -1, 4),  # loc preds
                    self.softmax(obm_conf.view(-1, self.num_classes)),  # conf preds
                )
        else:
            if self.use_refine:
                output = (
                    arm_loc.view(arm_loc.size(0), -1, 4),  # loc preds
                    tuple(arm_att_list),  # conf preds
                    obm_loc.view(obm_loc.size(0), -1, 4),  # loc preds
                    obm_conf.view(obm_conf.size(0), -1, self.num_classes),  # conf preds
                )
            else:
                output = (
                    obm_loc.view(obm_loc.size(0), -1, 4),  # loc preds
                    obm_conf.view(obm_conf.size(0), -1, self.num_classes),  # conf preds
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


def build_net(phase, size=320, num_classes=21, use_refine=False, dropout=1.0, residual=False, channel=False):
    if size not in [320, 512]:
        print("Error: Sorry only RefDetSSD320/512 is supported currently!")
        return

    return RefineSSD(size, num_classes=num_classes, use_refine=use_refine, phase=phase, dropout=dropout, residual=residual, channel=channel)
