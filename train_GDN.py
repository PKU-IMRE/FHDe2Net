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
def cal_psnr(src, tar, avg=False):

	data_range = 2
	diff = (src - tar)**2
	err = torch.sum(diff, (1,2,3)) / (src.shape[-1] * src.shape[-2] * src.shape[-3] )
	# err = criterion(src, tar)

	psnr = 10 * torch.log10((data_range ** 2) / err)
	if avg == False:
		return torch.sum(psnr)
	else:
		return torch.mean(psnr)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='my_loader_fs',  help='')
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
parser.add_argument('--annealEvery', type=int, default=150, help='epoch to reaching at learning rate of 0')
parser.add_argument('--lambdaIMG', type=float, default=1, help='lambdaIMG')
parser.add_argument('--lambdaCX', type=float, default=1, help='lambdaCX')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--netG', default='', help="path to netG1 (to continue training)")
parser.add_argument('--netGDN', default='', help="path to netGDN (to continue training)")
parser.add_argument('--netG2', default='', help="path to netG2 (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netD_moire', default='', help="path to netD_moire (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=500, help='interval for evauating(generating) images from valDataroot')
parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
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
                       split='train',
                       shuffle=True,
                       seed=opt.manualSeed,
                       pre=opt.pre)

# val_dataloader = getLoader(opt.dataset,
#                        opt.valDataroot,
#                        opt.originalSize_h,
#                        opt.originalSize_w,
#                        opt.imageSize_h,
#                        opt.imageSize_w,
#                        opt.valBatchSize,
#                        1,
#                        # mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
#                        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
#                        split='test',
#                        shuffle=False,
#                        seed=opt.manualSeed,
#                        pre=opt.pre)

print(len(dataloader))

# val_iterator = enumerate(val_dataloader, 0)
# val_cnt = 0

# get logger
trainLogger = open('%s/train.log' % opt.exp, 'a+')

inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize

# get models
netGDN = net.GDN()
print(netGDN)
visualizer = Visualizer(opt.display_port, opt.name)

if opt.netGDN != '':
  print("load pre-trained model!!!!!!!!!!!!!!!!!")
  netGDN.load_state_dict(torch.load(opt.netGDN))

netGDN.train()

vgg19 = VGG19()
vgg19.load_model(opt.vgg19)
vgg19.to(device)
utils.set_requires_grad(vgg19, False)
vgg19.eval()

vgg_layers = {'conv3_2':1}
criterionCAE = nn.L1Loss()
criterionCX = CXLoss(sigma=0.5, spatial_weight=0.5)
criterionCX_patch = CXLoss(sigma=0.5, spatial_weight=0.5)
netGDN.to(device)

criterionCAE.to(device)
criterionCX.to(device)
criterionCX_patch.to(device)

lambdaIMG = opt.lambdaIMG
lambdaCX = opt.lambdaCX

# get optimizer
my_lrG = opt.lrG - opt.epoch_count * (opt.lrG/opt.annealEvery)
optimizerG = optim.Adam(netGDN.parameters(), lr = my_lrG, betas = (opt.beta1, 0.999))

# NOTE training loop
ganIterations = 0
total_steps = 0
dataset_size = len(dataloader)

for epoch in range(opt.epoch_count, opt.annealEvery):
  trainLogger = open('%s/train.log' % opt.exp, 'a+')
  # switch the state !
  netGDN.train()

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
  for i, data in enumerate(dataloader, 0):

    netGDN.train()
    iter_start_time = time.time()
    if total_steps % 100 == 0:
        t_data = iter_start_time - iter_data_time
    visualizer.reset()
    total_steps += opt.batchSize
    epoch_iter += opt.batchSize

    input, target, name = data
    batch_size = target.size(0)
    input = input.cuda()
    target = target.cuda()

    optimizerG.zero_grad()
    oups = netGDN(input)
    vgg_target = vgg19(target)

    feat_target = vgg_target['conv3_2']
    CX_loss_list = [criterionCX(vgg19(x_hat)['conv3_2'] ,feat_target) for x_hat in oups]
    loss_CX = CX_loss_list[0]

    L = loss_CX
    L.backward()
    optimizerG.step()
    ganIterations += 1

    for i in range(x_hat.shape[0]):
        ccnt += 1
        ti1 = x_hat[i, :, :, :]
        tt1 = target[i, :, :, :]
        mi1 = util.my_tensor2im(ti1)
        mt1 = util.my_tensor2im(tt1)
        g_mi1 = cv2.cvtColor(mi1, cv2.COLOR_BGR2RGB)
        g_mt1 = cv2.cvtColor(mt1, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("res.jpg", mt1)
        my_psnr += Psnr(g_mt1, g_mi1)
        my_ssim_multi += ssim(g_mt1, g_mi1, multichannel=True)

    if total_steps % 100 == 0:

        current_visuals = OrderedDict([('input', input),
                                       ('output0', oups[0]),
                                       ('GT', target)
                                       ])

        losses = OrderedDict([
                              ('loss_CX', loss_CX.detach().cpu().float().numpy()),
                              ('my_psnr', my_psnr / (ccnt)), ('my_ssim_multi', my_ssim_multi / ccnt)])
        # print(losses)
        t = (time.time() - iter_start_time) / opt.batchSize
        trainLogger.write(visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data) + '\n')
        visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
        r = float(epoch_iter) / (dataset_size*opt.batchSize)
        if opt.display_port!=-1:
            visualizer.display_current_results(current_visuals, epoch, False)
            visualizer.plot_current_losses(epoch, r, opt, losses)
        netGDN.train()

  if epoch % 5 == 0:

        print('hit')
        my_file = open("./" + opt.name + "_" + "evaluation.txt", 'a+')
        torch.save(netGDN.state_dict(), '%s/netG_epoch_%d.pth' % (opt.exp, epoch))
        torch.save(netGDN.state_dict(), 'ckpt/netGDN.pth')
        # vcnt = 0
        # vpsnr = 0
        # vssim = 0
        # netGDN.eval()
        # for i, data in enumerate(val_dataloader, 0):
        #   input, target = data
        #   batch_size = target.size(0)
        #   input, target = input.to(device), target.to(device)
        #   x_hat = netGDN(input)
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
        # netGDN.train()


my_file.close()
trainLogger.close()