from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
# import torchvision.utils as vutils
from torch.autograd import Variable
from misc import *
import models.networks as net
from myutils.vgg16 import Vgg16
from myutils import utils
from visualizer import Visualizer
import time
import pdb
import torch.nn.functional as F
import sys
import os
from PIL import Image
import math
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as Psnr
import cv2
import util
from collections import OrderedDict

from model.vgg import VGG19
from model.generator import CXLoss

import models_metric
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='my_loader_LRN_f2_rand3',  help='')
parser.add_argument('--dataroot', required=False, default='', help='path to trn dataset')
parser.add_argument('--valDataroot', required=False, default='', help='path to val dataset')
parser.add_argument('--pre', type=str, default='', help='prefix of different dataset')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--epoch_count', type=int, default=1, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=1, help='input batch size')
parser.add_argument('--originalSize_h', type=int, default=539, help='the height / width of the original input image')
parser.add_argument('--originalSize_w', type=int, default=959, help='the height / width of the original input image')
parser.add_argument('--imageSize_h', type=int, default=512, help='the height / width of the cropped input image to network')
parser.add_argument('--imageSize_w', type=int, default=512, help='the height / width of the cropped input image to network')
parser.add_argument('--inputChannelSize', type=int, default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int, default=3, help='size of the output channels')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
parser.add_argument('--annealEvery', type=int, default=150, help='epoch to reaching at learning rate of 0')
parser.add_argument('--lambdaIMG', type=float, default=1, help='lambdaIMG')
parser.add_argument('--lambdaCX', type=float, default=1, help='lambdaCX')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')

parser.add_argument('--netGDN', default='', help="path to netGDN (to continue training)")
parser.add_argument('--netLRN', default='', help="path to netLRN (to continue training)")
parser.add_argument('--kernel_size', type=int, default=8, help='patch size for dct')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=500, help='interval for evauating(generating) images from valDataroot')
parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
parser.add_argument('--name', type=str, default='experiment_name',
                         help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--vgg19', default='./models/vgg19-dcbb9e9d.pth', help="path to vgg19.pth")
parser.add_argument('--list_file', type=str, default='', help='list_file')
parser.add_argument('--spatial_weight', type=float, default=0.5, help='spatial weight for CXloss')

opt = parser.parse_args()
print(opt)
opt.manualSeed = random.randint(1, 10000)
create_exp_dir(opt.exp)
device = torch.device("cuda:0")

# get dataloader
dataloader = getLoader(opt.dataset,
                       opt.dataroot,
                       opt.originalSize_h,
                       opt.originalSize_w,
                       opt.imageSize_h,
                       opt.imageSize_w,
                       opt.batchSize,
                       opt.workers,
                       # mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                       split='LRN_train_guide2',
                       shuffle=True,
                       seed=opt.manualSeed,
                       pre=opt.pre,
                       label_file='final_position.txt',
                       list_file=opt.list_file)

# val_dataloader = getLoader("my_loader_LSN",
#                        opt.dataroot,
#                        1024,
#                        1024,
#                        1024,
#                        1024,
#                        opt.batchSize,
#                        opt.workers,
#                        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
#                        split='LSN_test',
#                        shuffle=False,
#                        seed=opt.manualSeed,
#                        pre=opt.pre)

print(len(dataloader))

# val_iterator = enumerate(val_dataloader, 0)
# val_cnt = 0

# get logger
trainLogger = open('%s/train.log' % opt.exp, 'a+')

inputChannelSize = opt.inputChannelSize
outputChannelSize = opt.outputChannelSize

visualizer = Visualizer(opt.display_port, opt.name)

# # get models
netGDN = net.GDN()
if opt.netGDN != '':
  print("load pre-trained GSN model!!!!!!!!!!!!!!!!!")
  netGDN.load_state_dict(torch.load(opt.netGDN))
netGDN.eval()
utils.set_requires_grad(netGDN, False)

netLRN = net.LRN()
if opt.netLRN != '':
  print("load pre-trained LSN model!!!!!!!!!!!!!!!!!")
  netLRN.load_state_dict(torch.load(opt.netLRN))
netLRN.train()



# vgg = Vgg16()
# utils.init_vgg16('./models/')
# vgg.load_state_dict(torch.load(os.path.join('./models/', "vgg16.weight")))
# vgg.to(device)

vgg19 = VGG19()
vgg19.load_model(opt.vgg19)
vgg19.to(device)
utils.set_requires_grad(vgg19, False)
vgg19.eval()
# vgg_layers = ['conv3_3', 'conv4_2']
# vgg_layers = ['conv1_2', 'conv2_2', 'conv3_2']
# vgg_layers = {'conv1_2': 1.0, 'conv2_2': 1.0, 'conv3_2':0.5}
# vgg_layers = {'conv2_2': 1.0}
vgg_layers = {'conv3_2': 1}
criterionCAE = nn.L1Loss()
criterionCX = CXLoss(sigma=0.5, spatial_weight=opt.spatial_weight)
criterionCX_patch = CXLoss(sigma=0.5, spatial_weight=0.5)

netLRN.to(device)
netGDN.to(device)

criterionCAE.to(device)
criterionCX.to(device)
criterionCX_patch.to(device)

lambdaIMG = opt.lambdaIMG
lambdaCX = opt.lambdaCX

# get optimizer
my_lrG = opt.lrG - opt.epoch_count * (opt.lrG/opt.annealEvery)
optimizerG = optim.Adam( netLRN.parameters(), lr = my_lrG, betas = (opt.beta1, 0.999))
# NOTE training loop
ganIterations = 0
total_steps = 0
dataset_size = len(dataloader)
st = [0, 256, 512, 640]
patch_size = 384

net_metric = models_metric.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, spatial=False)
net_metric = net_metric.cuda()
utils.set_requires_grad(net_metric, requires_grad=False)
demoire_up_patch = torch.zeros([opt.batchSize, 3, patch_size, patch_size]).cuda()
res = 0
train_res = 0
for epoch in range(opt.epoch_count, opt.annealEvery):
  trainLogger = open('%s/train.log' % opt.exp, 'a+')
  # switch the state !
  netLRN.train()

  epoch_iter = 0
  epoch_start_time = time.time()
  iter_data_time = time.time()

  my_psnr = 0
  my_ssim = 0
  my_ssim_multi = 0

  adjust_learning_rate(optimizerG, opt.lrG, epoch, None, opt.annealEvery)
  print(50 * '-' + 'lr' + 50 * '-')
  print(str(optimizerG.param_groups[0]['lr']))
  print(50 * '-' + 'lr' + 50 * '-')

  ccnt = 0
  train_res = 0
  for i, data in enumerate(dataloader, 0):

    netLRN.train()
    iter_start_time = time.time()
    if total_steps % 100 == 0:
        t_data = iter_start_time - iter_data_time
    visualizer.reset()
    total_steps += opt.batchSize
    epoch_iter += opt.batchSize

    input_patch, target_patch, down_input, down_target, indexes1, indexes2 = data
    # print(indexes1.size())
    r = indexes1[:].numpy()
    c = indexes2[:].numpy()
    # print(r)
    batch_size = target_patch.size(0)

    # print(indexes)

    input_patch = input_patch.cuda()
    target_patch = target_patch.cuda()
    gray_input_patch = 0.299 * input_patch[:, 0, :, : ] + 0.587 * input_patch[:, 1, :, : ] + 0.114 * input_patch[:, 2, :, : ]
    gray_input_patch.unsqueeze_(1)
    gray_target_patch = 0.299 * target_patch[:, 0, :, : ] + 0.587 * target_patch[:, 1, :, : ] + 0.114 * target_patch[:, 2, :, : ]
    gray_target_patch.unsqueeze_(1)
    down_input = down_input.cuda()
    down_target = down_target.cuda()

    optimizerG.zero_grad()
    demoire_down = netGDN(down_input)[-1]
    demoire_up = F.interpolate(demoire_down, size=1024, mode='bilinear')
    down_target_up = F.interpolate(down_target, size=1024, mode='bilinear')
    # for i in range(batch_size):
    # print(demoire_up.shape)
    demoire_up_patch = demoire_up[:, :, indexes2[0]:indexes2[0] + patch_size, indexes1[0]: indexes1[0] + patch_size]
    down_target_up_patch = down_target_up[:, :, indexes2[0]:indexes2[0] + patch_size, indexes1[0]: indexes1[0] + patch_size]
    # demoire_up_patch.unsqueeze_(0)
    # down_target_up_patch.unsqueeze_(0)
    # print(demoire_up_patch.shape)
    x_hat_patch = netLRN(demoire_up_patch)

    vgg_x_hat = vgg19(x_hat_patch)
    vgg_target = vgg19(down_target_up_patch)

    loss_CX = 0.0
    if lambdaCX >0.0:
        for l, w in vgg_layers.items():
            loss_CX += w * criterionCX(vgg_x_hat[l], vgg_target[l])

        loss_CX = lambdaCX * loss_CX
    L = loss_CX

    L.backward()
    optimizerG.step()

    ganIterations += 1
    ccnt+=1

    if total_steps % 10 == 0:
        current_visuals = OrderedDict([
                                       ('down_input', down_input),
                                       ('demoire_down', demoire_down),
                                       ('down_target', down_target),
                                       ('input_patch', input_patch),
                                       ('demoire_up_patch', demoire_up_patch),
                                       ('x_hat_patch', x_hat_patch),
                                       ('down_target_up_patch', down_target_up_patch),
                                       ('target_patch', target_patch)
                                       # ('fake_moire', fake)
                                       ])

        losses = OrderedDict([
                              ('loss_CX', loss_CX.detach().cpu().float().numpy()),
                              # ('loss_perce', loss_perce.detach().cpu().float().numpy()),
                              # ('L_img2', L_img2.detach().cpu().float().numpy()),
                              # ('content_loss0', content_loss0.detach().cpu().float().numpy()),
                              # ('content_loss1', content_loss1.detach().cpu().float().numpy()),
                              # ('percep_metric', train_res.detach() .cpu().float().numpy()[0][0][0][0]/ (ccnt)),
                                ('my_psnr', my_psnr / (ccnt)), ('my_ssim_multi', my_ssim_multi / ccnt)])
        # print(losses)
        t = (time.time() - iter_start_time) / opt.batchSize
        trainLogger.write(visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data) + '\n')
        visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
        r = float(epoch_iter) / (dataset_size*opt.batchSize)
        if opt.display_port!=-1:
            visualizer.display_current_results(current_visuals, epoch, False)
            visualizer.plot_current_losses(epoch, r, opt, losses)

  if epoch % 5 == 0:
    my_file = open(opt.name + "/" + opt.name + "_" + "evaluation.txt", 'a+')
    print('hit')
    torch.save(netLRN.state_dict(), '%s/netLRN_epoch_%d.pth' % (opt.exp, epoch))
    # torch.save(netGDN.state_dict(), '%s/netGDN_epoch_%d.pth' % (opt.exp, epoch))
    # vcnt = 0
    # vpsnr = 0
    # vssim = 0
    # res = 0
    # netGDN.eval()
    # netLRN.eval()
    # utils.set_requires_grad(netGDN, False)
    # utils.set_requires_grad(netLRN, False)
    # for i, data in enumerate(val_dataloader, 0):
    #     if i % 50 == 0:
    #         print('testing: ' + str(i))
    #     input, target, down_input = data
    #     batch_size = input.size(0)
    #     input = input.cuda()
    #     target = target.cuda()
    #     down_input = down_input.cuda()
    #
    #     gray_input = 0.299 * input[:, 0, :, :] + 0.587 * input[:, 1, :, :] + 0.114 * input[:, 2, :, :]
    #     gray_input.unsqueeze_(1)
    #     gray_target = 0.299 * target[:, 0, :, :] + 0.587 * target[:, 1, :, :] + 0.114 * target[:, 2, :, :]
    #     gray_target.unsqueeze_(1)
    #
    #     demoire_down = netGDN(down_input)[-1].detach()
    #     demoire_up = F.interpolate(demoire_down, size=1024, mode='bilinear')
    #
    #     x_hat = netLRN(demoire_up)
    #     res += torch.sum(net_metric(target, x_hat).detach())
    #     vcnt += batch_size
    #     for i in range(x_hat.shape[0]):
    #         ti1 = x_hat[i, :, :, :]
    #         tt1 = target[i, :, :, :]
    #         mi1 = util.my_tensor2im(ti1)
    #         mt1 = util.my_tensor2im(tt1)
    #         g_mi1 = cv2.cvtColor(mi1, cv2.COLOR_BGR2RGB)
    #         g_mt1 = cv2.cvtColor(mt1, cv2.COLOR_BGR2RGB)
    #         vpsnr += Psnr(g_mt1, g_mi1)
    #         vssim += ssim(g_mt1, g_mi1, multichannel=True)
    #
    # my_file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '(' + str(epoch) + ')' + '\n')
    # my_file.write(str(float(res) / vcnt) + '-' + str(float(vpsnr) / vcnt) + '-' + str(float(vssim) / vcnt) + '\n')
    # my_file.close()
    # netGDN.train()
    # netLRN.train()
    # utils.set_requires_grad(netGDN, True)
    # utils.set_requires_grad(netLRN, True)

trainLogger.close()