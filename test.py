from __future__ import print_function
import argparse
import os
import sys
import random
import time
import pdb
from PIL import Image
import math
import numpy as np
import cv2
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as Psnr
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
cudnn.benchmark = True
cudnn.fastest = True

from misc import *
import models.networks as net
from myutils import utils
import models_metric

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='my_loader',  help='name for the dataset loader')
parser.add_argument('--dataroot', required=False,
  default='', help='path to trn dataset')
parser.add_argument('--netGDN', default='', help="path to netGDN")
parser.add_argument('--netLRN', default='', help="path to netLRN")
parser.add_argument('--netFDN', default='', help="path to netFDN")
parser.add_argument('--netFRN', default='', help="path to netFRN")
parser.add_argument('--kernel_size', type=int, default=8, help='patch size for dct')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--originalSize_h', type=int,
  default=539, help='the height of the original input image')
parser.add_argument('--originalSize_w', type=int,
  default=959, help='the height of the original input image')
parser.add_argument('--imageSize_h', type=int,
  default=512, help='the height of the cropped input image to network')
parser.add_argument('--imageSize_w', type=int,
  default=512, help='the width of the cropped input image to network')
parser.add_argument('--pre', type=str, default='', help='prefix of different dataset')
parser.add_argument('--image_path', type=str, default='', help='path to save the evaluated image')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--record', type=str, default='default.txt', help='file to record scores for each image')
parser.add_argument('--write', type=int, default=0, help='determine whether we save the result images')
opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

val_dataloader = getLoader(opt.dataset,
                       opt.dataroot,
                       opt.originalSize_h,
                       opt.originalSize_w,
                       opt.imageSize_h,
                       opt.imageSize_w,
                       opt.batchSize,
                       opt.workers,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                       split='test',
                       shuffle=False,
                       seed=opt.manualSeed,
                       pre=opt.pre)


if opt.write==0:
    print('no')
else:
    print('yes')

device = torch.device("cuda:0")

# dfine and load models 
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

netFDN = net.FDN(ORI_SIZE=opt.imageSize_w, KERNEL_SIZE=opt.kernel_size)
if opt.netFDN != '':
    print("load pre-trained FDN model!!!!!!!!!!!!!!!!!")
    netFDN.load_state_dict(torch.load(opt.netFDN))
netFDN.eval()
utils.set_requires_grad(netFDN, False)

netFRN = net.FRN()
if opt.netFRN != '':
    print("load pre-trained FRN model!!!!!!!!!!!!!!!!!")
    netFRN.load_state_dict(torch.load(opt.netFRN))
netFRN.eval()
utils.set_requires_grad(netFRN, False)

# load metric
net_metric = models_metric.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, spatial=False)
net_metric = net_metric.cuda()
utils.set_requires_grad(net_metric, requires_grad=False)

# to gpu
netLRN.to(device)
netGDN.to(device)
netFDN.to(device)
netFRN.to(device)

my_psnr = 0
my_ssim_multi = 0
patch_size = 384
res = 0
cnt1 = 0
f = open(opt.record, "w")
for i, data in enumerate(val_dataloader, 0):
    # netG.eval()
    print(50*'-')
    print(i)

    input, target, down_input, name= data
    batch_size = input.size(0)
    input = input.cuda()
    target = target.cuda()
    down_input = down_input.cuda()

    gray_input = 0.299 * input[:, 0, :, : ] + 0.587 * input[:, 1, :, : ] + 0.114 * input[:, 2, :, : ]
    gray_input.unsqueeze_(1)
    gray_target = 0.299 * target[:, 0, :, : ] + 0.587 * target[:, 1, :, : ] + 0.114 * target[:, 2, :, : ]
    gray_target.unsqueeze_(1)

    # GDN
    demoire_down = netGDN(down_input)[-1].detach()

    # upsampling
    demoire_up = F.interpolate(demoire_down, size=[opt.imageSize_h, opt.imageSize_w], mode='bilinear')

    # LRN
    demoire_up = netLRN(demoire_up)

    # get Y channel
    gray_demoire_up =  0.299 * demoire_up[:, 0, :, : ] + 0.587 * demoire_up[:, 1, :, : ] + 0.114 * demoire_up[:, 2, :, : ]
    gray_demoire_up.unsqueeze_(1)

    # FDN
    dct_oup = netFDN(gray_input, gray_demoire_up)

    # merge YUV from spatial and frequency domain
    demoire_up_u = -0.169 * demoire_up[:, 0, :, : ] - 0.331 * demoire_up[:, 1, :, : ] + 0.5 * demoire_up[:, 2, :, : ] - 1
    demoire_up_u.unsqueeze_(1)
    demoire_up_v = 0.5 * demoire_up[:, 0, :, : ] - 0.419 * demoire_up[:, 1, :, : ] - 0.081 * demoire_up[:, 2, :, : ] - 1
    demoire_up_v.unsqueeze_(1)
    yuv_merged_image = torch.cat([dct_oup, demoire_up_u, demoire_up_v], dim=1)

    # YUV to RGB
    r_merged_image = yuv_merged_image[:,0,:,:] + 1.403 * yuv_merged_image[:,2,:,:] + 1.403
    r_merged_image.unsqueeze_(1)
    g_merged_image = yuv_merged_image[:,0,:,:] -0.344 * yuv_merged_image[:,1,:,:] -0.714 * yuv_merged_image[:,2,:,:] -1.058
    g_merged_image.unsqueeze_(1)
    b_merged_image = yuv_merged_image[:,0,:,:] +1.773 * yuv_merged_image[:,1,:,:]  + 1.773
    b_merged_image.unsqueeze_(1)

    # FRN
    merged = torch.cat([r_merged_image, g_merged_image, b_merged_image], dim=1)
    x_hat = netFRN(merged) 

    # calculate scores
    cnt1+=batch_size
    tmp = torch.sum(net_metric(target, x_hat).detach())
    res += tmp
    L = str(tmp)
    print(res / cnt1)

    for j in range(x_hat.shape[0]):
        b, c, w, h = x_hat.shape
        ti1 = x_hat[j, :,:,: ]
        tt1 = target[j, :,:,: ]
        mi1 = cv2.cvtColor(utils.my_tensor2im(ti1), cv2.COLOR_BGR2RGB)
        mt1 = cv2.cvtColor(utils.my_tensor2im(tt1), cv2.COLOR_BGR2RGB)
        tmp2 =  Psnr(mt1, mi1)
        my_psnr += tmp2
        tmp3 = ssim(mt1, mi1, multichannel=True)
        my_ssim_multi += tmp3
        L = L +' ' + str(tmp2) +str(tmp3) + '\n'
        f.write(L)
        if opt.write == 1 and i<200/batch_size:
            if os.path.exists(opt.image_path) == False:
                os.makedirs(opt.image_path)
            cv2.imwrite(opt.image_path +os.sep+'res_' + name[j] +'.png', mi1)
    print(my_psnr / cnt1)
    print(my_ssim_multi / cnt1)

print("avergaed results:")
print(res / cnt1)
print(my_psnr / cnt1)
print(my_ssim_multi / cnt1)
f.close()
