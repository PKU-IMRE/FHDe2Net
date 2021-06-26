import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from NEDB import NEDB
from RNEDB import RNEDB
import functools
from torch.nn import init
from torchvision.models import vgg16
import numpy as np
from scipy.fftpack import dct as sci_dct


class SEBlock(nn.Module):
  def __init__(self, input_dim, reduction):
    super(SEBlock, self).__init__()
    mid = int(input_dim / reduction)
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
      nn.Linear(input_dim, reduction),
      nn.ReLU(inplace=True),
      nn.Linear(reduction, input_dim),
      nn.Sigmoid()
    )

  def forward(self, x):
    b, c, _, _ = x.size()
    y = self.avg_pool(x).view(b, c)
    y = self.fc(y).view(b, c, 1, 1)
    return x * y



class BottleneckBlock(nn.Module):
  def __init__(self, in_planes, out_planes, dropRate=0.0):
    super(BottleneckBlock, self).__init__()
    inter_planes = out_planes * 4
    self.relu = nn.ReLU(inplace=True)
    self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,padding=0, bias=False)
    self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,padding=1, bias=False)
    self.droprate = dropRate
  def forward(self, x):
    out = self.conv1(self.relu(x))
    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
    out = self.conv2(self.relu(out))
    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
    return torch.cat([x, out], 1)



class BottleneckBlock1(nn.Module):
  def __init__(self, in_planes, out_planes, dropRate=0.0):
    super(BottleneckBlock1, self).__init__()
    inter_planes = out_planes * 4
    self.relu = nn.ReLU(inplace=True)
    self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,padding=0, bias=False)
    self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=5, stride=1,padding=2, bias=False)
    self.droprate = dropRate
  def forward(self, x):
    out = self.conv1(self.relu(x))
    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
    out = self.conv2(self.relu(out))
    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
    return torch.cat([x, out], 1)


class BottleneckBlock2(nn.Module):
  def __init__(self, in_planes, out_planes, dropRate=0.0):
    super(BottleneckBlock2, self).__init__()
    inter_planes = out_planes * 4
    self.relu = nn.ReLU(inplace=True)
    self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,padding=0, bias=False)
    self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=7, stride=1,padding=3, bias=False)
    self.droprate = dropRate
  def forward(self, x):
    out = self.conv1(self.relu(x))
    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
    out = self.conv2(self.relu(out))
    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
    return torch.cat([x, out], 1)



class TransitionBlock(nn.Module):
  def __init__(self, in_planes, out_planes, dropRate=0.0):
    super(TransitionBlock, self).__init__()
    self.relu = nn.ReLU(inplace=True)
    self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,padding=0, bias=False)
    self.droprate = dropRate
  def forward(self, x):
    out = self.conv1(self.relu(x))
    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
    return F.upsample_nearest(out, scale_factor=2)



class TransitionBlock1(nn.Module):
  def __init__(self, in_planes, out_planes, dropRate=0.0):
    super(TransitionBlock1, self).__init__()
    self.relu = nn.ReLU(inplace=True)
    self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,padding=0, bias=False)
    self.droprate = dropRate
  def forward(self, x):
    out = self.conv1(self.relu(x))
    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
    return F.avg_pool2d(out, 2)



class TransitionBlock3(nn.Module):
  def __init__(self, in_planes, out_planes, dropRate=0.0):
    super(TransitionBlock3, self).__init__()
    self.relu = nn.ReLU(inplace=True)
    self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
    self.droprate = dropRate
  def forward(self, x):
    out = self.conv1(self.relu(x))
    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
    return out


# only for stride!=1 cases
def Division_map(ori_size, kernel, padding, stride):
  k1 = kernel
  s1 = 1
  p1 = kernel - padding  - 1
  final_size = 2 * p1 + ori_size + (ori_size-1) * (stride - 1)
  M = np.zeros([final_size, final_size])
  ret = np.zeros([(final_size - k1)/1 +1, (final_size - k1)/1 +1])

  i = p1
  j = p1
  margine = stride - 1
  while i <=final_size - p1:
    j = p1
    while j <= final_size - p1:
      M[i,j] = 1
      j+=(margine+1)
    i+=(margine+1)
  for i in range(ret.shape[0]):
    for j in range(ret.shape[1]):
      ret[i,j] = np.sum(M[i:i+kernel, j:j+kernel])
  return ret


def Division_map(ori_size1, ori_size2, kernel, padding, stride):
  k1 = kernel
  s1 = 1
  p1 = kernel - padding  - 1
  final_size1 = 2 * p1 + ori_size1 + (ori_size1-1) * (stride - 1)
  final_size2 = 2 * p1 + ori_size2 + (ori_size2-1) * (stride - 1)
  M = np.zeros([final_size1, final_size2])
  ret = np.zeros([(final_size1 - k1)/1 +1, (final_size2 - k1)/1 +1])

  i = p1
  j = p1
  margine = stride - 1
  while i <=final_size1 - p1:
    j = p1
    while j <= final_size2 - p1:
      M[i,j] = 1
      j+=(margine+1)
    i+=(margine+1)
  for i in range(ret.shape[0]):
    for j in range(ret.shape[1]):
      ret[i,j] = np.sum(M[i:i+kernel, j:j+kernel])
  return ret


def initDctCoeff2(kernel_size):
  D= sci_dct(np.eye(kernel_size), norm='ortho', axis = 0)
  dm = torch.FloatTensor(D)
  DCT = nn.Parameter(dm, requires_grad=False)
  iDCT = nn.Parameter(dm.t(), requires_grad=False)
  return DCT, iDCT


class dct_layer(nn.Module):

  def __init__(self, KERNEL_SIZE, STRIDE, PADDING, ORI_SIZE):
    super(dct_layer, self).__init__()
    self.KERNEL_SIZE = KERNEL_SIZE
    self.STRIDE = STRIDE
    self.PADDING = PADDING
    self.ORI_SIZE = ORI_SIZE
    self.CONV_SIZE = (ORI_SIZE + 2 * PADDING - KERNEL_SIZE ) // STRIDE + 1

    # image to patch
    self.my_conv = torch.nn.Conv2d(in_channels=1, out_channels=KERNEL_SIZE * KERNEL_SIZE, kernel_size=KERNEL_SIZE, stride=STRIDE, bias=False, padding=PADDING)
    conv_weight = np.zeros([KERNEL_SIZE * KERNEL_SIZE, 1, KERNEL_SIZE, KERNEL_SIZE], dtype=np.float32)
    for i in range(KERNEL_SIZE):
      for j in range(KERNEL_SIZE):
        conv_weight[i*KERNEL_SIZE+j, 0, i, j] = 1

    self.my_conv.weight.data.copy_(torch.from_numpy(conv_weight))
    self.my_conv.weight.requires_grad = False
  
    self.coeff_dct, self.coeff_idct = initDctCoeff2(KERNEL_SIZE)

  def forward(self, x):
    # convert image to patch
    b = self.my_conv(x)

    # DCT
    # convert from (B,C,W,H) to (B,W,H,KERNEL_SIZE, KERNEL_SIZE) to do dct
    bB, bC, bW, bH = b.shape
    b = torch.transpose(b, 1, 2)
    b = torch.transpose(b, 2, 3)
    b = b.view(bB, bW, bH, self.KERNEL_SIZE, self.KERNEL_SIZE)

    res_dct = torch.matmul(self.coeff_dct, b)
    res_dct = torch.matmul(res_dct, self.coeff_idct)

    # convert to pytorch formay for CNN operations
    res_dct = res_dct.view(bB, bW, bH, bC)
    res_dct = torch.transpose(res_dct, 2, 3)
    res_dct = torch.transpose(res_dct, 1, 2)

    return res_dct


class idct_layer(nn.Module):

  def __init__(self, KERNEL_SIZE, STRIDE, PADDING, ORI_SIZE):
    super(idct_layer, self).__init__()
    self.KERNEL_SIZE = KERNEL_SIZE
    self.STRIDE = STRIDE
    self.PADDING = PADDING
    self.ORI_SIZE = ORI_SIZE
    if ORI_SIZE == 1920:
      self.CONV_SIZE1 = (1080 + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1
      self.CONV_SIZE2 = (1920 + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1
      self.DM = Division_map(self.CONV_SIZE1, self.CONV_SIZE2, KERNEL_SIZE, PADDING, STRIDE)
      print(self.DM.shape)
      self.DM = torch.Tensor(self.DM.reshape(1, 1, 1080, 1920)).cuda()
    else:
      self.CONV_SIZE = (ORI_SIZE + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1
      self.DM = Division_map(self.CONV_SIZE, self.CONV_SIZE, KERNEL_SIZE, PADDING, STRIDE)
      self.DM = torch.Tensor(self.DM.reshape(1, 1, ORI_SIZE, ORI_SIZE)).cuda()

    # from patch to image
    self.my_deconv1 = torch.nn.ConvTranspose2d(in_channels=KERNEL_SIZE * KERNEL_SIZE, out_channels=1, kernel_size=KERNEL_SIZE, stride=STRIDE, bias=False, padding=PADDING)
    deconv_weight1 = np.zeros([KERNEL_SIZE * KERNEL_SIZE, 1, KERNEL_SIZE, KERNEL_SIZE], dtype=np.float32)
    for i in range(KERNEL_SIZE):
      for j in range(KERNEL_SIZE):
        deconv_weight1[i * KERNEL_SIZE + j, 0, i, j] = 1

    self.my_deconv1.weight.data.copy_(torch.from_numpy(deconv_weight1))
    self.my_deconv1.weight.requires_grad = False

    self.coeff_dct, self.coeff_idct = initDctCoeff2(KERNEL_SIZE)

  def forward(self, x):
    # IDCT
    # convert from (B,C,W,H) to (B,W,H,KERNEL_SIZE, KERNEL_SIZE) to do idct
    xB, xC, xW, xH = x.shape
    x = torch.transpose(x, 1, 2)
    x = torch.transpose(x, 2, 3)
    x = x.view(xB, xW, xH, self.KERNEL_SIZE, self.KERNEL_SIZE)

    # IDCT
    back = torch.matmul(self.coeff_idct, x)
    back = torch.matmul(back, self.coeff_dct)

    # convert to (B, C, W, H)
    back = back.view(xB, xW, xH, xC)
    back = torch.transpose(back, 2, 3)
    back = torch.transpose(back, 1, 2)

    # convert patch to image
    ret = torch.div(self.my_deconv1(back), self.DM)

    return ret



class GDN(nn.Module):
  def __init__(self):
    super(GDN, self).__init__()

    self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
    self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)

    self.dense_block1=BottleneckBlock2(64,64)
    self.trans_block1=TransitionBlock1(128,64)

    self.dense_block2=BottleneckBlock2(64,64)
    self.trans_block2=TransitionBlock1(128,64)

    self.dense_block3=BottleneckBlock2(64,64)
    self.trans_block3=TransitionBlock1(128,64)

    self.nb2 = NEDB(block_num=4, inter_channel=32, channel=64)

    self.dense_block4=BottleneckBlock2(64,64)
    self.trans_block4=TransitionBlock(128,64)

    self.dense_block5=BottleneckBlock2(128,64)
    self.trans_block5=TransitionBlock(192,64)

    self.dense_block6=BottleneckBlock2(128,64)
    self.trans_block6=TransitionBlock(192,64)

    self.tanh = nn.Tanh()
    self.relu = nn.LeakyReLU(0.2, inplace=True)

    self.conv61 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)



    self.final_conv = nn.Sequential(
      nn.Conv2d(64, 3, 3, 1, 1),
      nn.Tanh(),
    )

  def forward(self, x):

    old_state1 = None
    old_state2 = None
    oups = []

    for i in range(1):
      f0 = self.conv1(x)
      f1 = self.conv2(f0)
      x1 = self.dense_block1(f1)
      x1 = self.trans_block1(x1)

      x2 = (self.dense_block2(x1))
      x2 = self.trans_block2(x2)

      x3 = (self.dense_block3(x2))
      x3 = self.trans_block3(x3)

      x3, st1, st2 = self.nb2(x3, old_state1, old_state2)
      # x3, st = self.gru(x3, old_state)

      x4 = (self.dense_block4(x3))
      x4 = self.trans_block4(x4)

      x4 = torch.cat([x4, x2], 1)
      x5 = (self.dense_block5(x4))
      x5 = self.trans_block5(x5)

      x5 = torch.cat([x5, x1], 1)
      x6 = (self.dense_block6(x5))
      x6 = (self.trans_block6(x6))
      x61 = self.relu((self.conv61(x6)))

      x = self.final_conv(f0 + x61)
      oups.append(x)
      old_state1 = st1
      old_state2 = st2
    return oups


class LRN(nn.Module):
  def __init__(self):
    super(LRN, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
    self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)

    self.dense_block1 = BottleneckBlock2(64, 64)
    self.trans_block1 = TransitionBlock1(128, 64)

    ############# Block2-down 32-32  ##############
    self.dense_block2 = BottleneckBlock2(64, 64)
    self.trans_block2 = TransitionBlock1(128, 64)

    ############# Block3-down  16-16 ##############
    self.dense_block3 = BottleneckBlock2(64, 64)
    self.trans_block3 = TransitionBlock1(128, 64)

    ############# Block4-up  8-8  ##############
    self.dense_block4 = BottleneckBlock2(64, 64)
    self.trans_block4 = TransitionBlock(128, 64)

    ############# Block5-up  16-16 ##############
    self.dense_block5 = BottleneckBlock2(128, 64)
    self.trans_block5 = TransitionBlock(192, 64)

    self.dense_block6 = BottleneckBlock2(128, 64)
    self.trans_block6 = TransitionBlock(192, 64)

    self.tanh = nn.Tanh()
    self.relu=nn.LeakyReLU(0.2, inplace=True)


    self.final_conv = nn.Sequential(
      nn.Conv2d(64, 64, 3, 1, 1),
      nn.ReLU(),
      nn.Conv2d(64, 3, 3, 1, 1),
      nn.Tanh(),
    )

  def forward(self, x):
    ## 256x256
    f0 = self.conv1(x)
    f1 = self.conv2(f0)

    x1=self.dense_block1(f1)
    x1=self.trans_block1(x1)

    x2=(self.dense_block2(x1))
    x2=self.trans_block2(x2)

    x3=(self.dense_block3(x2))
    x3=self.trans_block3(x3)

    x4=(self.dense_block4(x3))
    x4=self.trans_block4(x4)

    # print(x4.shape)
    # print(x2.shape)
    x4_=torch.cat([x4,x2],1)
    x5=(self.dense_block5(x4_))
    x5=self.trans_block5(x5)

    x5_=torch.cat([x5,x1],1)
    x6=(self.dense_block6(x5_))
    x6=(self.trans_block6(x6))

    return  self.final_conv(f0 + x6)



class FDN(nn.Module):
  def __init__(self, KERNEL_SIZE=8, STRIDE=4, PADDING=2, ORI_SIZE=384):
    super(FDN, self).__init__()
    C = KERNEL_SIZE * KERNEL_SIZE
    self.dct_layer = dct_layer(KERNEL_SIZE, STRIDE, PADDING, ORI_SIZE)
    self.conv_merge = self.conv0 = nn.Sequential(
      nn.Conv2d(2 * C, C, 1, 1, 0),
      nn.PReLU(init=0.1)
    )
    self.se1 = SEBlock(C, C/2)
    self.conv0 = nn.Sequential(
      nn.Conv2d(C, 64, 3, 1, 1),
      nn.PReLU(init=0.1)
    )
    self.conv1 = nn.Sequential(
      nn.Conv2d(64, 64, 3, 1, 1),
      nn.PReLU(init=0.1)
    )
    self.conv2 = nn.Sequential(
      nn.Conv2d(64, 64, 3, 1, 1),
      nn.PReLU(init=0.1)
    )

    self.conv3 = nn.Sequential(
      nn.Conv2d(64, 64, 3, 1, 2, dilation=2),
      nn.PReLU(init=0.1)
    )
    self.conv3_1 =  nn.Sequential(
      nn.Conv2d(64, 64, 3, 1, 1),
      nn.PReLU(init=0.1)
    )
    self.conv3_2 =  nn.Sequential(
      nn.Conv2d(64, 64, 3, 1, 1),
      nn.PReLU(init=0.1)
    )
    self.conv3_3 =  nn.Sequential(
      nn.Conv2d(64, 64, 3, 1, 1),
      nn.PReLU(init=0.1)
    )

    self.conv4 = nn.Sequential(
      nn.Conv2d(64, 64, 3, 1, 4, dilation=4),
      nn.PReLU(init=0.1)
    )
    self.conv4_1 =  nn.Sequential(
      nn.Conv2d(64, 64, 3, 1, 1),
      nn.PReLU(init=0.1)
    )
    self.conv4_2 =  nn.Sequential(
      nn.Conv2d(64, 64, 3, 1, 1),
      nn.PReLU(init=0.1)
    )
    self.conv4_3 =  nn.Sequential(
      nn.Conv2d(64, 64, 3, 1, 1),
      nn.PReLU(init=0.1)
    )

    self.conv5 = nn.Sequential(
      nn.Conv2d(64, 64, 3, 1, 8, dilation=8),
      nn.PReLU(init=0.1)
    )
    self.conv5_1 =  nn.Sequential(
      nn.Conv2d(64, 64, 3, 1, 1),
      nn.PReLU(init=0.1)
    )
    self.conv5_2 =  nn.Sequential(
      nn.Conv2d(64, 64, 3, 1, 1),
      nn.PReLU(init=0.1)
    )
    self.conv5_3 =  nn.Sequential(
      nn.Conv2d(64, 64, 3, 1, 1),
      nn.PReLU(init=0.1)
    )

    self.conv6 = nn.Sequential(
      nn.Conv2d(64, 64, 3, 1, 16, dilation=16),
      nn.PReLU(init=0.1)
    )
    self.conv6_1 =  nn.Sequential(
      nn.Conv2d(64, 64, 3, 1, 1),
      nn.PReLU(init=0.1)
    )
    self.conv6_2 =  nn.Sequential(
      nn.Conv2d(64, 64, 3, 1, 1),
      nn.PReLU(init=0.1)
    )
    self.conv6_3 =  nn.Sequential(
      nn.Conv2d(64, 64, 3, 1, 1),
      nn.PReLU(init=0.1)
    )

    self.conv1x1 = nn.Sequential(
      nn.Conv2d(64 * 4, 64, 1, 1, 0),
      nn.PReLU(init=0.1)
    )
    self.conv7 = nn.Sequential(
      nn.Conv2d(64, 64, 3, 1, 1),
      nn.PReLU(init=0.1)
    )
    self.conv8 = nn.Sequential(
      nn.Conv2d(64, 32, 3, 1, 1),
      nn.PReLU(init=0.1)
    )
    self.conv9 = nn.Sequential(
      nn.Conv2d(32, C, 3, 1, 1),
    )
    self.idct_layer = idct_layer(KERNEL_SIZE, STRIDE, PADDING, ORI_SIZE)
  def forward(self, x, x2):
    x = self.dct_layer(x)
    x2 = self.dct_layer(x2)
    dct_in = x
    x = self.conv_merge(torch.cat([x, x2], dim=1))
    x = self.se1(x)
    x = self.conv0(x)
    x = self.conv1(x)
    F = self.conv2(x)
    
    x = self.conv3(F)
    x = self.conv3_1(x)
    x = self.conv3_2(x)
    x3 = self.conv3_3(x)

    x = self.conv4(F)
    x = self.conv4_1(x)
    x = self.conv4_2(x)
    x4 = self.conv4_3(x)

    x = self.conv5(F)
    x = self.conv5_1(x)
    x = self.conv5_2(x)
    x5 = self.conv5_3(x)

    x = self.conv6(F)
    x = self.conv6_1(x)
    x = self.conv6_2(x)
    x6 = self.conv6_3(x)
    
    x = self.conv1x1(torch.cat([x3,x4,x5,x6], dim=1))
    x = self.conv7(x)
    x = self.conv8(x)
    x = self.conv9(x)
    dct_out = x + dct_in
    res = self.idct_layer(dct_out)
    return res


class FRN(nn.Module):
  def __init__(self):
    super(FRN, self).__init__()

    self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
    # self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)

    self.relu = nn.LeakyReLU(0.2)

    self.dense1 = BottleneckBlock2(64, 64)
    self.trans1 = TransitionBlock3(128, 64)

    self.dense2 = BottleneckBlock2(64, 64)
    self.trans2 = TransitionBlock3(128, 64)

    self.dense3 = BottleneckBlock2(64, 64)
    self.trans3 = TransitionBlock3(128, 64)

    self.final_conv = nn.Sequential(
      nn.Conv2d(64, 3, 3, 1, 1),
      nn.Tanh(),
    )
  def forward(self, x):

    x0 =self.conv1(x)
    x1 = self.dense1(x0)
    x1 = self.trans1(x1)

    x2 = self.dense2(x1)
    x2 = self.trans2(x2)

    x3 = self.dense3(x2)
    x3 = self.trans3(x3)

    return self.final_conv(x0 + self.relu(x3))





