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
parser.add_argument('--dataset', required=False, default='my_loader_LRN_f2_rand2',  help='')
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

parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--annealEvery', type=int, default=150, help='epoch to reaching at learning rate of 0')
parser.add_argument('--lambdaIMG', type=float, default=1, help='lambdaIMG')
parser.add_argument('--lambdaCX', type=float, default=1, help='lambdaCX')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')

parser.add_argument('--netGDN', default='', help="path to netGDN (to continue training)")
parser.add_argument('--netLRN', default='', help="path to netLRN (to continue training)")
parser.add_argument('--netFDN', default='', help="path to netFDN (to continue training)")
parser.add_argument('--netFRN', default='', help="path to netFRN (to continue training)")
parser.add_argument('--kernel_size', type=int, default=8, help='patch size for dct')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=500, help='interval for evauating(generating) images from valDataroot')
parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
parser.add_argument('--name', type=str, default='experiment_name',
                         help='name of the experiment. It decides where to store samples and models')

parser.add_argument('--vgg19', default='./models/vgg19-dcbb9e9d.pth', help="path to vgg19.pth")
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
                       split='LRN_train_guide',
                       shuffle=True,
                       seed=opt.manualSeed,
                       pre=opt.pre,
                       label_file='final_position.txt')

# val_dataloader = getLoader("my_loader_LRN",
#                        opt.dataroot,
#                        1024,
#                        1024,
#                        1024,
#                        1024,
#                        opt.batchSize,
#                        opt.workers,
#                        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
#                        split='LRN_test',
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

# get models
netGDN = net.GDN()
if opt.netGDN != '':
  print("load pre-trained GDN model!!!!!!!!!!!!!!!!!")
  netGDN.load_state_dict(torch.load(opt.netGDN))
netGDN.eval()
utils.set_requires_grad(netGDN, False)

netLRN = net.LRN()
if opt.netLRN != '':
  print("load pre-trained LRN model!!!!!!!!!!!!!!!!!")
  netLRN.load_state_dict(torch.load(opt.netLRN))
netLRN.eval()
utils.set_requires_grad(netLRN, False)

netFDN = net.FDN(opt.kernel_size)
if opt.netFDN != '':
  print("load pre-trained FDN model!!!!!!!!!!!!!!!!!")
  netFDN.load_state_dict(torch.load(opt.netFDN))
netFDN.train()

netFRN = net.FRN()
if opt.netFRN != '':
  print("load pre-trained Refine model!!!!!!!!!!!!!!!!!")
  netFRN.load_state_dict(torch.load(opt.netFRN))
netFRN.train()

vgg19 = VGG19()
vgg19.load_model(opt.vgg19)
vgg19.to(device)
utils.set_requires_grad(vgg19, False)
vgg19.eval()

vgg_layers = {'conv3_2': 1}
criterionCAE = nn.L1Loss()
criterionCX = CXLoss(sigma=0.5, spatial_weight=0.5)
criterionCX_patch = CXLoss(sigma=0.5, spatial_weight=0.5)

netLRN.to(device)
netGDN.to(device)
netFDN.to(device)
netFRN.to(device)

criterionCAE.to(device)
criterionCX.to(device)
criterionCX_patch.to(device)

lambdaIMG = opt.lambdaIMG
lambdaCX = opt.lambdaCX

# get optimizer
my_lrG = opt.lrG - opt.epoch_count * (opt.lrG/opt.annealEvery)
optimizerG = optim.Adam(itertools.chain(netFDN.parameters(), netFRN.parameters()), lr = my_lrG, betas = (opt.beta1, 0.999))

# NOTE training loop
ganIterations = 0
total_steps = 0
dataset_size = len(dataloader)
st = [0, 256, 512, 640]
patch_size = 384

net_metric = models_metric.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, spatial=False)
net_metric = net_metric.cuda()
utils.set_requires_grad(net_metric, requires_grad=False)

res = 0
train_res = 0
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay+1):
  trainLogger = open('%s/train.log' % opt.exp, 'a+')
  # switch the state !
  netFDN.train()

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

    netFDN.train()
    iter_start_time = time.time()
    if total_steps % 100 == 0:
        t_data = iter_start_time - iter_data_time
    visualizer.reset()
    total_steps += opt.batchSize
    epoch_iter += opt.batchSize

    input_patch, target_patch, down_input, down_target, indexes1, indexes2 = data
    r = indexes1[:].numpy()
    c = indexes2[:].numpy()
    batch_size = target_patch.size(0)

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
    demoire_up = netLRN(demoire_up)

    demoire_up_patch = demoire_up[:, :, indexes2[0]:indexes2[0] + patch_size, indexes1[0]: indexes1[0] + patch_size]
    gray_demoire_up_patch =  0.299 * demoire_up_patch[:, 0, :, : ] + 0.587 * demoire_up_patch[:, 1, :, : ] + 0.114 * demoire_up_patch[:, 2, :, : ]
    gray_demoire_up_patch.unsqueeze_(1)
    dct_oup = netFDN(gray_input_patch, gray_demoire_up_patch)

    demoire_up_patch_u = -0.169 * demoire_up_patch[:, 0, :, : ] - 0.331 * demoire_up_patch[:, 1, :, : ] + 0.5 * demoire_up_patch[:, 2, :, : ] - 1
    demoire_up_patch_u.unsqueeze_(1)
    demoire_up_patch_v = 0.5 * demoire_up_patch[:, 0, :, : ] - 0.419 * demoire_up_patch[:, 1, :, : ] - 0.081 * demoire_up_patch[:, 2, :, : ] - 1
    demoire_up_patch_v.unsqueeze_(1)
    yuv_merged_image = torch.cat([dct_oup, demoire_up_patch_u, demoire_up_patch_v], dim=1)
    r_merged_image = yuv_merged_image[:,0,:,:] + 1.403 * yuv_merged_image[:,2,:,:] + 1.403
    r_merged_image.unsqueeze_(1)
    g_merged_image = yuv_merged_image[:,0,:,:] -0.344 * yuv_merged_image[:,1,:,:] -0.714 * yuv_merged_image[:,2,:,:] -1.058
    g_merged_image.unsqueeze_(1)
    b_merged_image = yuv_merged_image[:,0,:,:] +1.773 * yuv_merged_image[:,1,:,:]  + 1.773
    b_merged_image.unsqueeze_(1)
    merged_patch = torch.cat([r_merged_image, g_merged_image, b_merged_image], dim=1)
    x_hat = netFRN(merged_patch)

    dct_oup = dct_oup.repeat(1, 3, 1, 1)
    gray_target_patch = gray_target_patch.repeat(1, 3, 1, 1)

    vgg_x_hat = vgg19(dct_oup)
    vgg_target = vgg19(gray_target_patch)
    loss_FDN_perce = criterionCAE(vgg_x_hat['conv1_2'], vgg_target['conv1_2']) + criterionCAE(vgg_x_hat['conv2_2'], vgg_target['conv2_2'])

    vgg_x_hat2 = vgg19(x_hat)
    vgg_target2 = vgg19(target_patch)
    loss_CX = 0.0
    if lambdaCX >0.0:
        for l, w in vgg_layers.items():
            loss_CX += w * criterionCX(vgg_x_hat2[l], vgg_target2[l])

        loss_CX = lambdaCX * loss_CX
    L = loss_CX + loss_FDN_perce

    L.backward()
    optimizerG.step()

    ganIterations += 1
    train_res += torch.sum(net_metric(target_patch, x_hat).detach())

    for i in range(x_hat.shape[0]):
        ccnt += 1
        ti1 = x_hat[i, :, :, :]
        tt1 = target_patch[i, :, :, :]
        mi1 = util.my_tensor2im(ti1)
        mt1 = util.my_tensor2im(tt1)
        g_mi1 = cv2.cvtColor(mi1, cv2.COLOR_BGR2RGB)
        g_mt1 = cv2.cvtColor(mt1, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("res.jpg", mt1)
        my_psnr += Psnr(g_mt1, g_mi1)
        my_ssim_multi += ssim(g_mt1, g_mi1, multichannel=True)



    if total_steps % 10 == 0:

        current_visuals = OrderedDict([
                                       ('input_patch', input_patch),
                                       ('demoire_up_patch', demoire_up_patch),
                                       ('dct_oup', dct_oup),
                                       ('merged_patch', merged_patch),
                                       ('x_hat', x_hat),
                                       ('target_patch', target_patch)
                                       ])

        losses = OrderedDict([
                              ('L', L.detach().cpu().float().numpy()),
                              # ('loss_l1', L_img.detach().cpu().float().numpy()),
                              # ('L_img2', L_img2.detach().cpu().float().numpy()),
                              # ('content_loss0', content_loss0.detach().cpu().float().numpy()),
                              # ('content_loss1', content_loss1.detach().cpu().float().numpy()),
                              ('train_res', train_res.detach() .cpu().float().numpy()/ (ccnt)),
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

        print('hit')
        torch.save(netFDN.state_dict(), '%s/netFDN_epoch_%d.pth' % (opt.exp, epoch))
        torch.save(netFRN.state_dict(),'%s/netFRN_epoch_%d.pth' % (opt.exp, epoch)) 

        torch.save(netFDN.state_dict(), 'ckpt/netFDN.pth')
        torch.save(netFRN.state_dict(),'ckpt/netFDN.pth') 
        # vcnt = 0
        # vpsnr = 0
        # vssim = 0
        # netG.eval()
        # for i, data in enumerate(val_dataloader, 0):
        #   input, target = data
        #   batch_size = target.size(0)
        #   input, target = input.to(device), target.to(device)
        #   f1, x_hat = netG(input)
        #   for i in range(x_hat.shape[0]):
        #       vcnt += 1
        #       ti1 = x_hat[i, :, :, :]
        #       tt1 = target[i, :, :, :]
        #       mi1 = util.my_tensor2im(ti1)
        #       mt1 = util.my_tensor2im(tt1)
        #       g_mi1 = cv2.cvtColor(mi1, cv2.COLOR_BGR2RGB)
        #       g_mt1 = cv2.cvtColor(mt1, cv2.COLOR_BGR2RGB)
        #       vpsnr += Psnr(g_mt1, g_mi1)
        #
        # my_file.write(str(epoch) + str('-') + str(total_steps) + '\n')
        # my_file.write(str(float(vpsnr) / vcnt) + '\n')
        # print("val:")
        # print(float(vpsnr) / vcnt)
        # trainLogger.close()
        # netG.train()


my_file.close()
trainLogger.close()