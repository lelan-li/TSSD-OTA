import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import v2, v1, AnnotationTransform, BaseTransform, VOCDetection, detection_collate, seq_detection_collate, VOCroot, VIDroot, VOC_CLASSES, VID_CLASSES
from utils.augmentations import SSDAugmentation, seqSSDAugmentation
from layers.modules import MultiBoxLoss, seqMultiBoxLoss, AttentionLoss
from ssd import build_ssd
import numpy as np
import time

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc_512.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--resume_from_ssd', default='./weights/ssd300_VIDDET_512/ssd300_VIDDET_5000.pth', type=str, help='Resume vgg and extras from ssd checkpoint')
parser.add_argument('--freeze', default=0, type=int, help='Freeze, 1. vgg, extras; 2. vgg, extras, conf, loc')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=120000, type=int, help='Number of training iterations')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=True, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder', default='./weights/test/', help='Location to save checkpoint models')
parser.add_argument('--dataset_name', default='seqVID2017', help='Which dataset')
parser.add_argument('--step_list', nargs='+', type=int, default=[30,50], help='step_list for learning rate')
parser.add_argument('--ssd_dim', default=300, type=int, help='ssd_dim 300 or 512')
parser.add_argument('--gpu_ids', default='1,0', type=str, help='gpu number')
parser.add_argument('--augm_type', default='seqssd', type=str, help='how to transform data')
parser.add_argument('--tssd',  default='tblstm', type=str, help='ssd or tssd')
parser.add_argument('--seq_len', default=4, type=int, help='Batch size for training')
parser.add_argument('--set_file_name',  default='train_video_remove_no_object', type=str, help='train set name')
parser.add_argument('--attention', default=True, type=str2bool, help='attention')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

cfg = (v1, v2)[args.version == 'v2']

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset_name=='VOC0712':
    train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    num_classes = len(VOC_CLASSES) + 1
    data_root = VOCroot
# elif (args.dataset_name=='VID2017'):
#     train_sets = 'train'
#     num_classes = len(VID_CLASSES) + 1
#     data_root = VIDroot
elif args.dataset_name=='VID2017':
    train_sets = 'train'
    num_classes = len(VID_CLASSES) + 1
    data_root = VIDroot
else:
    train_sets = 'train_remove_noobject'
    num_classes = len(VID_CLASSES) + 1
    data_root = VIDroot

set_filename = args.set_file_name
collate_fn = seq_detection_collate if args.dataset_name=='seqVID2017' else detection_collate

ssd_dim = args.ssd_dim  # only support 300 now
means = (104, 117, 123)
mean_np = np.array(means, dtype=np.int32)

batch_size = args.batch_size
# accum_batch_size = 32
# iter_size = accum_batch_size / batch_size
weight_decay = args.weight_decay
stepvalues = args.step_list
max_iter = args.step_list[-1]
gamma = 0.1
momentum = args.momentum

if args.visdom:
    import visdom
    viz = visdom.Visdom()

ssd_net = build_ssd('train', ssd_dim, num_classes, tssd=args.tssd, attention=args.attention)
net = ssd_net

if args.cuda:
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)

elif args.resume_from_ssd != 'ssd':
    from collections import OrderedDict
    print('training from pretrained vgg and extras, loading {}...'.format(args.resume_from_ssd))
    ssd_weights = torch.load(args.resume_from_ssd)
    ssd_vgg_weights = OrderedDict()
    ssd_extras_weights = OrderedDict()
    ssd_loc_weights = OrderedDict()
    ssd_conf_weights = OrderedDict()
    for key, weight in ssd_weights.items():
        key_split = key.split('.')
        subnet_name = key_split[0]
        if subnet_name == 'vgg':
            ssd_vgg_weights[key_split[1] + '.' + key_split[2]] = weight
        elif subnet_name == 'extras':
            ssd_extras_weights[key_split[1] + '.' + key_split[2]] = weight
        elif subnet_name == 'loc':
            ssd_loc_weights[key_split[1] + '.' + key_split[2]] = weight
        elif subnet_name == 'conf':
            ssd_conf_weights[key_split[1] + '.' + key_split[2]] = weight
    ssd_net.vgg.load_state_dict(ssd_vgg_weights)
    ssd_net.extras.load_state_dict(ssd_extras_weights)
    ssd_net.loc.load_state_dict(ssd_loc_weights)
    ssd_net.conf.load_state_dict(ssd_conf_weights)

else:
    vgg_weights = torch.load(args.save_folder + '/../'+ args.basenet)# + '/../'
    print('Loading base network...')
    ssd_net.vgg.load_state_dict(vgg_weights)

if args.freeze:
    if args.freeze == 1:
        print('Freeze vgg, extras')
        freeze_nets = [ssd_net.vgg, ssd_net.extras]
    elif args.freeze == 2:
        print('Freeze vgg, extras, conf, loc')
        freeze_nets = [ssd_net.vgg, ssd_net.extras, ssd_net.conf, ssd_net.loc]
    elif args.freeze == 3:
        print('Freeze vgg, extras, attention')
        freeze_nets = [ssd_net.vgg, ssd_net.extras, ssd_net.attention]
    else:
        freeze_nets = []
    for freeze_net in freeze_nets:
        for param in freeze_net.parameters():
            param.requires_grad = False

if args.cuda:
    net = net.cuda()

def xavier(param):
    init.xavier_uniform(param)

def orthogonal(param):
    init.orthogonal(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def conv_weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        # m.bias.data.zero_()

def orthogonal_weights_init(m):
    if isinstance(m, nn.Conv2d):
        orthogonal(m.weight.data)
        m.bias.data.fill_(1)

if not args.resume:
    if args.resume_from_ssd != 'ssd':
        # print('Initializing Multibox weights...')
        # ssd_net.loc.apply(weights_init)
        # ssd_net.conf.apply(weights_init)
        if args.attention:
            print('Initializing Attention weights...')
            ssd_net.attention.apply(conv_weights_init)
        if args.tssd in ['lstm', 'edlstm', 'gru', 'tblstm',  'tbedlstm', 'outlstm']:
            print('Initializing RNN weights...')
            ssd_net.rnn.apply(orthogonal_weights_init)

    else:
        print('Initializing extra, loc, conf, weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
        if args.tssd in ['lstm', 'edlstm', 'gru', 'tblstm','tbedlstm', 'outlstm']:
            print('Initializing RNN weights...')
            ssd_net.rnn.apply(orthogonal_weights_init)
        if args.attention:
            print('Initializing Attention weights...')
            ssd_net.attention.apply(conv_weights_init)

if args.augm_type == 'ssd':
    data_transform = SSDAugmentation
elif args.augm_type == 'seqssd':
    data_transform = seqSSDAugmentation
else:
    data_transform = BaseTransform

# att_params = list(map(id, net.module.attention.parameters()))
# base_params = filter(lambda p: id(p) not in att_params, net.module.parameters())

if args.tssd in ['lstm', 'edlstm', 'gru', 'tblstm', 'tbedlstm', 'outlstm']:
    optimizer = optim.SGD(net.module.attention.parameters()
                               # [{'params': net.module.loc.parameters()},
                               # {'params': net.module.conf.parameters()}]
                               # {'params': net.module.loc.parameters()}]
                              ,lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_rnn = optim.RMSprop(net.module.rnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = seqMultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)
else:
    if args.freeze == 0:
        optimizer = optim.SGD( net.parameters()
            , lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.freeze == 1:
        optimizer = optim.SGD(#net.parameters()
                          [{'params': net.module.attention.parameters(), 'lr':args.lr*10},
                           {'params': net.module.loc.parameters()},
                           {'params': net.module.conf.parameters()}]
                          #  {'params': net.module.loc.parameters()}]
                          , lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer_att = optim.SGD(net.parameters()
    criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)
if args.attention:
    att_criterion = AttentionLoss(args.ssd_dim)

# optimizer = optim.RMSprop(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def train():
    net.train()
    epoch = 0
    print('Loading Dataset...')
    dataset = VOCDetection(data_root, train_sets, data_transform(
        ssd_dim, means), AnnotationTransform(dataset_name=args.dataset_name),
                           dataset_name=args.dataset_name, set_file_name=set_filename, seq_len=args.seq_len)

    epoch_size = len(dataset) // args.batch_size

    print('Training TSSD on', dataset.name, ', how many videos:', len(dataset), ', sequence length:', args.seq_len) if args.tssd in ['lstm', 'edlstm', 'outlstm'] else print('Training SSD on', dataset.name)
    print('lr:',args.lr , 'steps:', stepvalues, 'max_liter:', max_iter)
    step_index = 0
    if args.visdom:
        # initialize visdom loss plot
        lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 4)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Att Loss','Loss'],
            )
        )
    batch_iterator = None
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=collate_fn, pin_memory=True)

    for iteration in range(args.start_iter, max_iter):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
        if iteration in stepvalues:
            step_index += 1
            # adjust_learning_rate(optimizer_rnn, args.gamma, step_index)
            adjust_learning_rate(optimizer, args.gamma, step_index)
            epoch += 1

        images, targets, masks = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            masks = Variable(masks.cuda())
            targets = [[Variable(seq_anno.cuda(), volatile=True) for seq_anno in batch_anno] for batch_anno in targets] if args.dataset_name=='seqVID2017' \
               else [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            masks = Variable(masks)
            targets = [[Variable(seq_anno, volatile=True) for seq_anno in batch_anno] for batch_anno in targets] if args.dataset_name=='seqVID2017' \
                else [Variable(anno, volatile=True) for anno in targets]

        # forward
        t0 = time.time()
        out, att = net(images)
        if args.tssd != 'ssd':
            optimizer_rnn.zero_grad()
        optimizer.zero_grad()
        if args.attention:
            loss_att, upsampled_att_map = att_criterion(att,masks)
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c + 0.5*loss_att
        else:
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
        loss.backward()
        if args.tssd != 'ssd':
            nn.utils.clip_grad_norm(net.module.rnn.parameters(), 5)
            optimizer_rnn.step()
        optimizer.step()
        t1 = time.time()

        if iteration % 10 == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
            if args.visdom and args.send_images_to_visdom:
                random_batch_index = np.random.randint(images.size(0))
                if images.dim() == 5:
                    for time_idx, time_step in enumerate([0,-1]):
                        img_viz = (images.data[random_batch_index,time_step].cpu().numpy().transpose(1,2,0) + mean_np).transpose(2,0,1)
                        viz.image(img_viz, win=120+time_idx, opts=dict(title='seq2_frame_%s' % time_step))
                        for scale, att_map_viz in enumerate(upsampled_att_map[time_step]):
                            viz.heatmap(att_map_viz[random_batch_index, 0, :, :].data.cpu().numpy()[::-1],
                                        win=130*(time_idx+1) + scale,
                                        opts=dict(title='seq2_attmap_time%s_scale%s' % (time_step,scale), colormap='Jet'))
                        viz.heatmap(masks[random_batch_index, time_step, 0, :, :].data.cpu().numpy()[::-1],
                                    win=180 + time_idx,
                                    opts=dict(title='seq2_attmap_gt_%s' % time_step, colormap='Jet'))
                else:
                    img_viz = (images.data[random_batch_index].cpu().numpy().transpose(1,2,0) + mean_np).transpose(2,0,1)
                    viz.image(img_viz, win=1, opts=dict(title='ssd_frame_gt', colormap='Jet'))
                    for scale, att_map_viz in enumerate(upsampled_att_map):
                        viz.heatmap(att_map_viz[random_batch_index, 0, :, :].data.cpu().numpy()[::-1], win=2+scale,
                            opts=dict(title='ssd_attmap_%s' % scale, colormap='Jet'))
                    viz.heatmap(masks[random_batch_index, 0, :, :].data.cpu().numpy()[::-1], win=2 + len(upsampled_att_map),
                            opts=dict(title='ssd_attmap_gt', colormap='Jet'))

        if args.visdom:
            if iteration == 1000:
                # initialize visdom loss plot
                lot = viz.line(
                    X=torch.zeros((1,)).cpu(),
                    Y=torch.zeros((1, 4)).cpu(),
                    opts=dict(
                        xlabel='Iteration',
                        ylabel='Loss',
                        title='Current SSD Training Loss',
                        legend=['Loc Loss', 'Conf Loss', 'Att Loss', 'Loss'],
                    )
                )
            viz.line(
                X=torch.ones((1, 4)).cpu() * iteration,
                Y=torch.Tensor([loss_l.data[0], loss_c.data[0], loss_att.data[0],
                    loss.data[0]]).unsqueeze(0).cpu(),
                win=lot,
                update='append'
            )

        if iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), args.save_folder+'ssd'+ str(ssd_dim) + '_' + args.dataset_name + '_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(), args.save_folder+'ssd'+ str(ssd_dim) + '_' + args.dataset_name + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    # lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] *= gamma


if __name__ == '__main__':
    train()
