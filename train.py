import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import AnnotationTransform, BaseTransform, VOCDetection, MOTDetection, detection_collate, seq_detection_collate, mb_cfg, dataset_training_cfg

from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss, AttentionLoss, RefineMultiBoxLoss
from layers.functions import PriorBox
import numpy as np
import time
import logging

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def print_log(args):
    logging.info('model_name: '+ args.model_name)
    logging.info('ssd_dim: '+ str(args.ssd_dim))
    logging.info('Backbone: '+ args.backbone)
    if 'RefineDet' in args.backbone:
        logging.info('Refine: ' + str(args.refine))
        logging.info('Dropout: ' + str(args.drop))
    logging.info('attention: '+ str(args.attention))
    if args.attention:
        logging.info('residual attention: ' + str(args.res_attention))
        logging.info('channel attention: ' + str(args.channel_attention))
    logging.info('Predection model: '+ str(args.pm))
    if args.resume:
        logging.info('resume: '+ args.resume )
        logging.info('start_iter: '+ str(args.start_iter))
    elif args.resume_from_ssd != 'ssd':
        logging.info('resume_from_ssd: '+ args.resume_from_ssd )
    else:
        logging.info('load pre-trained backbone: '+ args.basenet )
    logging.info('freeze: '+ str(args.freeze))
    logging.info('lr: '+ str(args.lr))
    logging.info('gamam: '+ str(args.gamma))
    logging.info('step_list: '+ str(args.step_list))
    logging.info('save_interval: '+ str(args.save_interval))
    logging.info('dataset_name: '+ args.dataset_name )
    logging.info('set_file_name: '+ args.set_file_name )
    logging.info('gpu_ids: '+ args.gpu_ids)
    logging.info('augm_type: '+ args.augm_type)
    logging.info('batch_size: '+ str(args.batch_size))
    # logging.info('tssd: '+ args.tssd )
    # if args.tssd != 'ssd':
    #     logging.info('seq_len: '+ str(args.seq_len))
    #     logging.info('skip: '+ str(args.skip))
    #     logging.info('association: '+ str(args.association))
    #     if args.association:
    #         logging.info('asso_top_k: '+ str(args.asso_top_k))
    #         logging.info('asso_conf: '+ str(args.asso_conf))
    logging.info('loss weights: '+ str(args.loss_coe))

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint') #'./weights/tssd300_VID2017_b8s8_RSkipTBLstm_baseAugmDrop2Clip5_FixVggExtraPreLocConf/ssd300_seqVID2017_20000.pth'
parser.add_argument('--resume_from_ssd', default='ssd', type=str, help='Resume vgg and extras from ssd checkpoint')
parser.add_argument('--freeze', default=0, type=int, help='Freeze, 1. vgg, extras; 2. vgg, extras, conf, loc; 3. vgg, extras, rnn, attention, conf, loc')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder', default='./weights040/test', help='Location to save checkpoint models')
parser.add_argument('--dataset_name', default='VOC0712', help='VOC0712/VIDDET/seqVID2017/MOT17Det/seqMOT17Det')
parser.add_argument('--step_list', nargs='+', type=int, default=[30,50], help='step_list for learning rate')
parser.add_argument('--backbone', default='RefineDet_VGG', type=str, help='Backbone')
parser.add_argument('--pm', default=0.0, type=float, help='use predection model or not, the float denotes the channel increment')
parser.add_argument('--refine', default=True, type=str2bool, help='Only work when backbone==RefineDet')
parser.add_argument('--drop', default=1.0, type=float, help='DropOut, Only work when backbone==RefineDet')
parser.add_argument('--model_name', default='ssd', type=str, help='which model selected')
parser.add_argument('--ssd_dim', default=320, type=int, help='ssd_dim 300, 320 or 512')
parser.add_argument('--gpu_ids', default='3,2', type=str, help='gpu number')
parser.add_argument('--augm_type', default='base', type=str, help='how to transform data')
# parser.add_argument('--tssd',  default='ssd', type=str, help='ssd or tssd')
# parser.add_argument('--seq_len', default=8, type=int, help='sequence length for training')
parser.add_argument('--set_file_name',  default='train', type=str, help='train_VID_DET/train_video_remove_no_object/train, MOT dataset does not use it')
parser.add_argument('--attention', default=True, type=str2bool, help='add attention module')
parser.add_argument('--res_attention', default=False, type=str2bool, help='add attention module')
parser.add_argument('--channel_attention', default=True, type=str2bool, help='add attention module')
# parser.add_argument('--association', default=False, type=str2bool, help='dynamic set prior box through time')
# parser.add_argument('--asso_top_k', default=1, type=int, help='top_k for association loss')
# parser.add_argument('--asso_conf', default=0.1, type=float, help='conf thresh for association loss')
parser.add_argument('--loss_coe', nargs='+', type=float, default=[1.0,1.0, 0.5], help='coefficients for loc, conf, att, asso')
# parser.add_argument('--skip', default=False, type=str2bool, help='select sequence data in a skip way')
parser.add_argument('--bn', default=False, type=str2bool, help='select sequence data in a skip way')
parser.add_argument('--save_interval', default=5000, type=int, help='frequency of checkpoint saving')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
current_time = time.strftime("%b_%d_%H:%M:%S_%Y", time.localtime())
logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename=os.path.join(args.save_folder, current_time+'.log'),
                filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
# logging.info(args)
print_log(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

if args.dataset_name in ['MOT15', 'seqMOT15']:
    prior = 'MOT_300'
    cfg = mb_cfg[prior]
else:
    prior = 'VOC_'+ str(args.ssd_dim)
    if args.ssd_dim==512 and 'RefineDet' in args.backbone:
        prior += '_RefineDet'
    cfg = mb_cfg[prior]

train_sets, num_classes, data_root = dataset_training_cfg[args.dataset_name]

set_filename = args.set_file_name
collate_fn = seq_detection_collate if args.dataset_name[:3]=='seq' else detection_collate

ssd_dim = args.ssd_dim  # only support 300 now
means = (104, 117, 123)
mean_np = np.array(means, dtype=np.int32)

batch_size = args.batch_size
weight_decay = args.weight_decay
stepvalues = args.step_list
max_iter = args.step_list[-1]
gamma = 0.1
momentum = args.momentum

if args.visdom:
    import visdom
    viz = visdom.Visdom()

if args.backbone == 'RFB_VGG':
    from model.rfbnet_vgg import build_net
    ssd_net = build_net('train', ssd_dim, num_classes)
elif 'RefineDet' in args.backbone:
    if args.attention:
        from model.attrefinedet_vgg import build_net
        ssd_net = build_net('train', size=ssd_dim, num_classes=num_classes, use_refine=args.refine, dropout=args.drop, residual=args.res_attention, channel=args.channel_attention)
    else:
        from model.refinedet_vgg import build_net
        ssd_net = build_net('train', size=ssd_dim, num_classes=num_classes, use_refine=args.refine)
elif 'ResNet' in args.backbone:
    from model.ssd_resnet import build_net
    ssd_net = build_net('train', backbone=args.backbone, size=ssd_dim, num_classes=num_classes, prior=prior, pm=args.pm)
else:
    from model.ssd import build_net
    ssd_net = build_net('train', ssd_dim, num_classes, tssd=args.tssd, attention=args.attention, prior=prior, bn=args.bn)
                   # single_batch=int(args.batch_size/len(args.gpu_ids.split(','))))
net = ssd_net

# if args.cuda and torch.cuda.is_available():
if device==torch.device('cuda'):
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True

net = net.to(device)

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
elif args.resume_from_ssd != 'ssd':
    from collections import OrderedDict
    print('training from pretrained backbone and extras, loading {}...'.format(args.resume_from_ssd))
    ssd_weights = torch.load(args.resume_from_ssd)
    ssd_backbone_weights = OrderedDict()
    ssd_extras_weights = OrderedDict()
    ssd_loc_weights = OrderedDict()
    ssd_conf_weights = OrderedDict()
    for key, weight in ssd_weights.items():
        key_split = key.split('.')
        subnet_name = key_split[0]
        if subnet_name == 'backbone':
            ssd_backbone_weights[key_split[1] + '.' + key_split[2]] = weight
        elif subnet_name == 'extras':
            ssd_extras_weights[key_split[1] + '.' + key_split[2]] = weight
        elif subnet_name == 'loc':
            ssd_loc_weights[key_split[1] + '.' + key_split[2]] = weight
        elif subnet_name == 'conf':
            ssd_conf_weights[key_split[1] + '.' + key_split[2]] = weight
    ssd_net.backbone.load_state_dict(ssd_backbone_weights)
    ssd_net.extras.load_state_dict(ssd_extras_weights)
    ssd_net.loc.load_state_dict(ssd_loc_weights)
    ssd_net.conf.load_state_dict(ssd_conf_weights)
else:
    backbone_weights = torch.load(args.save_folder + '/../'+ args.basenet)
    print('Loading base network...')
    ssd_net.backbone.load_state_dict(backbone_weights)

if args.freeze:
    if args.freeze == 1:
        print('Freeze backbone, extras')
        freeze_nets = [ssd_net.backbone, ssd_net.extras]
    elif args.freeze == 2:
        print('Freeze backbone, extras, conf, loc')
        freeze_nets = [ssd_net.backbone, ssd_net.extras, ssd_net.conf, ssd_net.loc]
    else:
        freeze_nets = []
    for freeze_net in freeze_nets:
        for param in freeze_net.parameters():
            param.requires_grad = False

if not args.resume:
    from model.networks import net_init
    net_init(ssd_net, args.backbone, resume_from_ssd=args.resume_from_ssd, attention=args.attention, pm=args.pm, refine=args.refine)

if args.augm_type == 'ssd':
    data_transform = SSDAugmentation
else:
    data_transform = BaseTransform

# optimize
if args.freeze == 1:
    optimizer = optim.SGD([{'params': net.module.attention.parameters(), 'lr':args.lr*10},
                           {'params': net.module.loc.parameters()},
                           {'params': net.module.conf.parameters()}],
                           lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
else:
    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# criterion
if 'RefineDet' in args.backbone and args.refine:
    use_refine = True
    arm_criterion = RefineMultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, device=device, only_loc=True)
    criterion = RefineMultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, object_score = 0.01, device=device)
else:
    use_refine = False
    criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, device=device)

if args.attention:
    att_criterion = AttentionLoss(args.ssd_dim)

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward().to(device)

def train():
    net.train()
    epoch = 0
    print('Loading Dataset...' + args.dataset_name)
    if args.dataset_name in ['MOT15', 'seqMOT15', 'MOT17Det', 'seqMOT17Det']:
        dataset = MOTDetection(data_root, train_sets, data_transform(
            ssd_dim, means),dataset_name=args.dataset_name)
    else:
        dataset = VOCDetection(data_root, train_sets, data_transform(ssd_dim, means),
                               AnnotationTransform(dataset_name=args.dataset_name),
                               dataset_name=args.dataset_name, set_file_name=set_filename,
                               use_mask=args.attention)

    epoch_size = len(dataset) // args.batch_size

    print('Training SSD on', dataset.name, 'dataset size:', len(dataset), '\n',
          'lr:',args.lr , 'steps:', stepvalues, 'max_liter:', max_iter)
    step_index = 0
    if args.visdom:
        # initialize visdom loss plot
        y_dim = 3
        legend = ['Loss', 'Loc Loss', 'Conf Loss',]
        if use_refine:
            y_dim += 1
            legend += ['Arm Loc Loss',]
            if not args.attention:
                y_dim += 1
                legend += ['Arm conf Loss',]
        if args.attention:
            y_dim += 1
            legend += ['Att Loss',]
        lot = viz.line(
            X=torch.zeros((1.,)),
            Y=torch.zeros((1., y_dim)),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title=args.save_folder.split('/')[-1],
                legend=legend,
            )
        )
    batch_iterator = None
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=collate_fn, pin_memory=True)

    for iteration in range(args.start_iter, max_iter+1):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
        if iteration in stepvalues:
            step_index += 1
            # adjust_learning_rate(optimizer_rnn, args.gamma, step_index)
            adjust_learning_rate(optimizer, args.gamma, step_index)
            epoch += 1
        collected_data = next(batch_iterator)
        with torch.no_grad():
            images, targets = collected_data[:2]
            images = images.to(device)
            targets = [anno.to(device) for anno in targets]
            if args.attention:
                masks = collected_data[2].to(device)

        # forward
        t0 = time.time()
        loss = torch.tensor(0., requires_grad=True).to(device)
        out = net(images)
        # backward
        optimizer.zero_grad()
        if use_refine:
            loss_arm_l = arm_criterion(out[0], priors, targets)
            loss_l, loss_c = criterion(out[2:], priors, targets, arm_data=out[:2])
            # if args.attention:
            loss += args.loss_coe[0] * loss_arm_l
            if args.attention:
                att_maps = out[1]
            # else:
            #     loss_arm_l, loss_arm_c = arm_criterion(out[:2], priors, targets)
            #     loss += args.loss_coe[0] * loss_arm_l #+  args.loss_coe[1] * loss_arm_c
        else:
            loss_l, loss_c = criterion(out, priors, targets)
        loss += args.loss_coe[0] * loss_l + args.loss_coe[1] * loss_c
        if args.attention:
            loss_att, upsampled_att_map = att_criterion(att_maps, masks)
            loss += args.loss_coe[2]*loss_att

        loss.backward()
        optimizer.step()
        t1 = time.time()

        if iteration % 10 == 0:
            logging.info('iter ' + repr(iteration) + '||Loss: %.4f, lr: %.5f||Timer: %.4f sec.' % (loss, optimizer.param_groups[0]['lr'], t1 - t0))
            if args.visdom and args.send_images_to_visdom:
                random_batch_index = np.random.randint(images.size(0))
                img_viz = (images[random_batch_index].cpu().numpy().transpose(1,2,0) + mean_np).transpose(2,0,1)
                viz.image(img_viz, win=20, opts=dict(title='ssd2_frame_gt', colormap='Jet'))
                for scale, att_map_viz in enumerate(upsampled_att_map):
                    viz.heatmap(att_map_viz[random_batch_index, 0, :, :].detach().cpu().numpy()[::-1], win=21+scale,
                        opts=dict(title='ssd2_attmap_%s' % scale, colormap='Jet'))
                viz.heatmap(masks[random_batch_index, 0, :, :].cpu().numpy()[::-1], win=21 + len(upsampled_att_map),
                        opts=dict(title='ssd2_attmap_gt', colormap='Jet'))

        if args.visdom:
            y_dis = [loss.cpu(), args.loss_coe[0]*loss_l.cpu(), args.loss_coe[1]*loss_c.cpu()]
            if iteration == 1000:
                # initialize visdom loss plot
                lot = viz.line(
                    X=torch.zeros((1.,)),
                    Y=torch.zeros((1., y_dim)),
                    opts=dict(
                        xlabel='Iteration',
                        ylabel='Loss',
                        title=args.save_folder.split('/')[-1],
                        legend=legend,
                    )
                )
            if use_refine:
                if args.attention:
                    y_dis += [args.loss_coe[0]*loss_arm_l.cpu(),]
                else:
                    y_dis += [args.loss_coe[0]*loss_arm_l.cpu(), args.loss_coe[1]*loss_arm_c.cpu()]
            if args.attention:
                y_dis += [args.loss_coe[2]*loss_att.cpu(),]
            # update = 'append' if iteration
            viz.line(
                X=torch.ones((1., y_dim)) * iteration,
                Y=torch.FloatTensor(y_dis).unsqueeze(0),
                win=lot,
                update='append',
                opts=dict(
                    xlabel='Iteration',
                    ylabel='Loss',
                    title=args.save_folder.split('/')[-1],
                    legend=legend,)
            )

        if iteration % args.save_interval == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, args.model_name+ str(ssd_dim) + '_' + args.dataset_name + '_' +
                       repr(iteration) + '.pth'))
    torch.save(ssd_net.state_dict(),
               os.path.join(args.save_folder, args.model_name + str(ssd_dim) + '_' + args.dataset_name + '_' +
                            repr(iteration) + '.pth'))
    print('Complet Training. Saving state, iter:', iteration)

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] *= gamma


if __name__ == '__main__':
    train()
