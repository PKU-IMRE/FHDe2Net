import torch
import torch.nn as nn
import torch.nn.functional as F


class CXLoss(nn.Module):

    def __init__(self, sigma=0.1, b=1.0, similarity="consine", spatial_weight=0.5):
        super(CXLoss, self).__init__()
        self.similarity = similarity
        self.sigma = sigma
        self.b = b
        self.spatial_weight = spatial_weight

    def center_by_T(self, featureI, featureT):
        # Calculate mean channel vector for feature map.
        meanT = featureT.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        return featureI - meanT, featureT - meanT

    def l2_normalize_channelwise(self, features):
        # Normalize on channel dimension (axis=1)
        norms = features.norm(p=2, dim=1, keepdim=True)
        features = features.div(norms)
        return features

    def compute_meshgrid(self, shape):
        N, C, H, W = shape
        rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
        cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

        feature_grid = torch.meshgrid(rows, cols)
        feature_grid = torch.stack(feature_grid).unsqueeze(0)
        feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

        return feature_grid

    def create_using_L2(self, I_features, T_features):
        """
        Calculating the distance between each feature of I and T
        :param I_features:
        :param T_features:
        :return: raw_distance: [N, C, H, W, H*W], each element of which is the distance between I and T at each position
        """
        assert I_features.size() == T_features.size()
        N, C, H, W = I_features.size()

        Ivecs = I_features.view(N, C, -1)
        Tvecs = T_features.view(N, C, -1)
        #
        square_I = torch.sum(Ivecs*Ivecs, dim=1, keepdim=False)
        square_T = torch.sum(Tvecs*Tvecs, dim=1, keepdim=False)
        # raw_distance
        raw_distance = []
        for i in range(N):
            Ivec, Tvec, s_I, s_T = Ivecs[i, ...], Tvecs[i, ...], square_I[i, ...], square_T[i, ...]
            # matrix multiplication
            # print(Ivec.permute(1, 0).shape)
            # print(Tvec.shape)
            AB = torch.mm(Ivec.permute(1, 0), Tvec)
            dist = s_I.view(-1, 1) + s_T.view(1, -1) - 2*AB

            raw_distance.append(dist.view(1, H, W, H*W).permute(0, 3, 1, 2))
        raw_distance = torch.cat(raw_distance, dim=0)
        raw_distance = torch.clamp(raw_distance, 0.0)


        relative_dist = self.calc_relative_distances(raw_distance)

        CX = self.calc_CX(relative_dist)
        return CX


    def create_using_dotP(self, I_features, T_features):
        I_features, T_features = self.center_by_T(I_features, T_features)

        I_features = self.l2_normalize_channelwise(I_features)
        T_features = self.l2_normalize_channelwise(T_features)

        dist = []
        N = T_features.size()[0]
        for i in range(N):
            # NCHW
            featureT_i = T_features[i, :, :, :].unsqueeze(0)
            # NCHW
            featureI_i = I_features[i, :, :, :].unsqueeze(0)
            featureT_patch = self.patch_decomposition(featureT_i)
            # Calculate cosine similarity
            # See the torch document for functional.conv2d
            dist_i = F.conv2d(featureI_i, featureT_patch)
            dist.append(dist_i)

        # NCHW
        dist = torch.cat(dist, dim=0)

        raw_distance = (1. - dist) / 2.

        relative_dist = self.calc_relative_distances(raw_distance)

        CX = self.calc_CX(relative_dist)

        return CX

    def create(self, I_features, T_features):
        # spatial loss
        grid = self.compute_meshgrid(I_features.shape).cuda()
        cx_sp = self.create_using_L2(grid, grid)

        # feature loss
        cx_feat = self.create_using_dotP(I_features, T_features)
        return cx_sp, cx_feat


    def patch_decomposition(self, features):
        N, C, H, W = features.shape
        assert N == 1
        P = H * W
        # NCHW --> 1x1xCxHW --> HWxCx1x1
        patches = features.view(1, 1, C, P).permute((3, 2, 0, 1))
        return patches

    def calc_relative_distances(self, raw_dist, axis=1):
        epsilon = 1e-5
        div = torch.min(raw_dist, dim=axis, keepdim=True)[0]
        relative_dist = raw_dist / (div + epsilon)
        return relative_dist

    def calc_CX(self, dist, axis=1):
        W = torch.exp((self.b - dist) / self.sigma)
        W_sum = W.sum(dim=axis, keepdim=True)
        return W.div(W_sum)

    def forward(self, I_features, T_features):

        cx_sp, cx_feat = self.create(I_features, T_features)
        CX = (1. - self.spatial_weight) * cx_feat + self.spatial_weight * cx_sp

        CX = CX.max(dim=3)[0].max(dim=2)[0]
        CX = CX.mean(1)
        CX = -torch.log(CX)
        CX = torch.mean(CX)
        return CX


def init_params(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Down(nn.Module):

    def __init__(self, size, in_channels, out_channels):
        super(Down, self).__init__()
        self.size = size
        self.features = [nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
                         nn.LayerNorm([out_channels, size, size]),
                         nn.LeakyReLU(0.2),
                         nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
                         nn.LayerNorm([out_channels, size, size]),
                         nn.LeakyReLU(0.2)]
        self.features = nn.Sequential(*self.features)
        self.upsample = nn.Upsample(size=(self.size, self.size), mode='bilinear')

    def forward(self, image, x=None):
        out = self.upsample(image)
        if x is not None:
            out = torch.cat([x, out], dim=1)

        return self.features(out)


class Generator(nn.Module):

    def __init__(self, image_size):
        super(Generator, self).__init__()

        self.image_size = image_size

        self.image_sizes = [256, 128, 64, 32, 16, 8, 4]

        self.in_dims = [259, 515, 515, 515, 515, 515, 3]
        self.out_dims = [256, 256, 512, 512, 512, 512, 512]

        self.rec = nn.Sigmoid()
        self.conv = nn.Conv2d(256, 3, kernel_size=(1, 1))
        self.down1 = Down(self.image_sizes[0], 259, 256)
        self.donw2 = Down(self.image_sizes[1], 515, 256)
        self.down3 = Down(self.image_sizes[2], 515, 512)
        self.down4 = Down(self.image_sizes[3], 515, 512)
        self.down5 = Down(self.image_sizes[4], 515, 512)
        self.down6 = Down(self.image_sizes[5], 515, 512)
        self.down7 = Down(self.image_sizes[6], 3, 512)

        init_params(self.modules())

    def forward(self, x):
        down7 = self.down7(x)
        down7 = F.interpolate(down7, size=(self.image_sizes[-2], self.image_sizes[-2]), mode='bilinear')
        down6 = self.down6(x, down7)
        down6 = F.interpolate(down6, size=(self.image_sizes[-3], self.image_sizes[-3]), mode='bilinear')
        down5 = self.down5(x, down6)
        down5 = F.interpolate(down5, size=(self.image_sizes[-4], self.image_sizes[-4]), mode='bilinear')
        down4 = self.down4(x, down5)
        down4 = F.interpolate(down4, size=(self.image_sizes[-5], self.image_sizes[-5]), mode='bilinear')
        down3 = self.down3(x, down4)
        down3 = F.interpolate(down3, size=(self.image_sizes[-6], self.image_sizes[-6]), mode='bilinear')
        down2 = self.donw2(x, down3)
        down2 = F.interpolate(down2, size=(self.image_sizes[-7], self.image_sizes[-7]), mode='bilinear')
        down1 = self.down1(x, down2)
        return (self.conv(down1) + 1.) / 2.


# class Down(nn.Module):
#
#     def __init__(self, in_channels, out_channels, BN=False, IN=True):
#         super(Down, self).__init__()
#         modules = [nn.LeakyReLU(0.2, inplace=False),
#                   nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
#         if BN:
#             modules.append(nn.BatchNorm2d(out_channels))
#         if IN:
#             modules.append(nn.InstanceNorm2d(out_channels))
#
#         self.feature = nn.Sequential(*modules)
#         init_params(self.feature)
#
#     def forward(self, x):
#         return self.feature(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, BN=False, IN=True, dropout=True):
        super(Up, self).__init__()
        modules = [nn.LeakyReLU(0.2, inplace=False),
                   nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if BN:
            modules.append(nn.BatchNorm2d(out_channels))
        if IN:
            modules.append(nn.InstanceNorm2d(out_channels))
        if dropout:
            modules.append(nn.Dropout(0.5))

        self.feature = nn.Sequential(*modules)

    def forward(self, c1, c2=None):
        c1 = self.feature(c1)
        if c2 is not None:
            return torch.cat([c1, c2], dim=1)
        else:
            return c1


# class Generator(nn.Module):
#
#     def __init__(self, out_channels=64):
#         super(Generator, self).__init__()
#         # 256 x 256 x 6
#         self.down1 = nn.Conv2d(6, out_channels, kernel_size=4, stride=2, padding=1)
#         # 128 x 128 x 64
#         self.down2 = Down(out_channels, out_channels * 2)
#         # 64 x 64 x 128
#         self.down3 = Down(out_channels * 2, out_channels * 2)
#         # 32 x 32 x 256
#         self.down4 = Down(out_channels * 2, out_channels * 2)
#         # 16 x 16 x 512
#         self.down5 = Down(out_channels * 2, out_channels * 2)
#         # 8 x 8 x 512
#         self.down6 = Down(out_channels * 2, out_channels * 2)
#         # 4 x 4 x 512
#         self.down7 = Down(out_channels * 2, out_channels * 2)
#         # 2 x 2 x 512
#         self.down8 = Down(out_channels * 2, out_channels * 2, IN=False)
#
#         # 1 x 1 x 512
#         self.up1 = Up(out_channels * 2, out_channels * 2)
#         # 2 x 2 x (512 + 512)
#         self.up2 = Up(out_channels * 2 * 2, out_channels * 2)
#         # 4 x 4 x (512 + 512)
#         self.up3 = Up(out_channels * 2 * 2, out_channels * 2)
#         # 8 x 8 x (512 + 512)
#         self.up4 = Up(out_channels * 2 * 2, out_channels * 2, dropout=False)
#         # 16 x 16 x (512 + 512)
#         self.up5 = Up(out_channels * 2 * 2, out_channels * 2, dropout=False)
#         # 32 x 32 x (256 + 256)
#         self.up6 = Up(out_channels * 2 * 2, out_channels * 2, dropout=False)
#         # 64 x 64 x (128 + 128)
#         self.up7 = Up(out_channels * 2 * 2, out_channels, dropout=False)
#         # 128 x 128 x (64 + 64)
#         self.up8 = Up(out_channels * 2, 3, IN=False, dropout=False)
#         # 256 x 256 x 3
#         self.rec = nn.Sigmoid()
#
#     def forward(self, s, t):
#         '''
#         :param dImage: degraded image
#         :param wImage: wrap guidance image
#         :return:
#         '''
#         x = torch.cat([s, t], dim=1)
#         down1 = self.down1(x)
#         down2 = self.down2(down1)
#         down3 = self.down3(down2)
#         down4 = self.down4(down3)
#         down5 = self.down5(down4)
#         down6 = self.down6(down5)
#         down7 = self.down7(down6)
#         down8 = self.down8(down7)
#
#         up1 = self.up1(down8, down7)
#         up2 = self.up2(up1, down6)
#         up3 = self.up3(up2, down5)
#         up4 = self.up4(up3, down4)
#         up5 = self.up5(up4, down3)
#         up6 = self.up6(up5, down2)
#         up7 = self.up7(up6, down1)
#         up8 = self.up8(up7)
#         rec = self.rec(up8)
#         return rec
