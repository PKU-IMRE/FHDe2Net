import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init

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




class ConvGRU(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super(ConvGRU, self).__init__()
        pad_x = int(dilation * (kernel - 1) / 2)
        self.conv_xz = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xr = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xn = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)

        pad_h = int((kernel - 1) / 2)
        self.conv_hz = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hr = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hn = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)

        # self.nl = NEDB(inter_channel=oup_dim / 2, channel=oup_dim)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, h=None):
        if h is None:
            z = F.sigmoid(self.conv_xz(x))
            f = F.tanh(self.conv_xn(x))
            h = z * f
        else:
            # h.unsqueeze_(1)
            z = F.sigmoid(self.conv_xz(x) + self.conv_hz(h))
            r = F.sigmoid(self.conv_xr(x) + self.conv_hr(h))
            n = F.tanh(self.conv_xn(x) + self.conv_hn(r * h))
            h = (1 - z) * h + z * n

        h = self.relu(h)
        return h, h

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels


        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        self.se0 = SEBlock(self.inter_channels, self.inter_channels/2)
        self.se1 = SEBlock(self.inter_channels, self.inter_channels/2)
        self.se2 = SEBlock(self.inter_channels, self.inter_channels/2)
        self.se3 = SEBlock(self.inter_channels, self.inter_channels/2)
        self.gru1 = ConvGRU(self.inter_channels, self.inter_channels, 1, 1)
        self.gru2 = ConvGRU(self.inter_channels, self.inter_channels, 1, 1)
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.theta_ = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))
    def forward(self, x, corr1, corr2):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)
        # print(x.shape)
        g_x, ret_corr1 = self.gru1(self.se0(self.g(x)), corr1)
        g_x = g_x.view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x, ret_corr2 = self.gru2(self.se1(self.theta(x)), corr2)
        theta_x = theta_x.view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        # print(theta_x.shape)
        phi_x = self.se3(self.phi(x))
        phi_x = phi_x.view(batch_size, self.inter_channels, -1)
        # print(phi_x.shape)
        f = torch.matmul(theta_x, phi_x)

        theta_x_ = self.theta_(x)
        theta_x_ = self.se3(theta_x_)
        theta_x_ = theta_x_.view(batch_size, self.inter_channels, -1)
        theta_x_ = theta_x_.permute(0, 2, 1)


        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y + theta_x_
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z, ret_corr1, ret_corr2


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)
