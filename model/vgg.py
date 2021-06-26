import torch.nn as nn
import torch


def conv2d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True))


def conv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
        self.conv1_1 = nn.Sequential(conv(3, 64), nn.ReLU())
        self.conv1_2 = nn.Sequential(conv(64, 64), nn.ReLU())
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.conv2_1 = nn.Sequential(conv(64, 128), nn.ReLU())
        self.conv2_2 = nn.Sequential(conv(128, 128), nn.ReLU())
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.conv3_1 = nn.Sequential(conv(128, 256), nn.ReLU())
        self.conv3_2 = nn.Sequential(conv(256, 256), nn.ReLU())
        self.conv3_3 = nn.Sequential(conv(256, 256), nn.ReLU())
        self.conv3_4 = nn.Sequential(conv(256, 256), nn.ReLU())
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.conv4_1 = nn.Sequential(conv(256, 512), nn.ReLU())
        self.conv4_2 = nn.Sequential(conv(512, 512), nn.ReLU())

    def load_model(self, model_file):
        vgg19_dict = self.state_dict()
        pretrained_dict = torch.load(model_file)
        vgg19_keys = vgg19_dict.keys()
        pretrained_keys = pretrained_dict.keys()
        for k, pk in zip(vgg19_keys, pretrained_keys):
            vgg19_dict[k] = pretrained_dict[pk]
        self.load_state_dict(vgg19_dict)

    # def forward(self, input_images):
    #     # print(self.mean)
    #     # input_images = (input_images - self.mean) / self.std
    #     feature = {}
    #     feature['conv1_1'] = self.conv1_1(input_images)
    #     feature['conv1_2'] = self.conv1_2(feature['conv1_1'])
    #     feature['pool1'] = self.pool1(feature['conv1_2'])
    #     feature['conv2_1'] = self.conv2_1(feature['pool1'])
    #     feature['conv2_2'] = self.conv2_2(feature['conv2_1'])
    #     feature['pool2'] = self.pool2(feature['conv2_2'])
    #     feature['conv3_1'] = self.conv3_1(feature['pool2'])
    #     feature['conv3_2'] = self.conv3_2(feature['conv3_1'])
    #     feature['conv3_3'] = self.conv3_3(feature['conv3_2'])
    #     feature['conv3_4'] = self.conv3_4(feature['conv3_3'])
    #     feature['pool3'] = self.pool3(feature['conv3_4'])
    #     feature['conv4_1'] = self.conv4_1(feature['pool3'])
    #     feature['conv4_2'] = self.conv4_2(feature['conv4_1'])
    #
    #     return feature
    def forward(self, input_images):
        # print(self.mean)
        # input_images = (input_images - self.mean) / self.std
        feature = {}
        tmp = self.conv1_1(input_images)
        tmp = self.conv1_2(tmp)
        feature['conv1_2'] = tmp
        tmp = self.pool1(tmp)
        tmp = self.conv2_1(tmp)
        tmp = self.conv2_2(tmp)
        feature['conv2_2'] = tmp
        tmp = self.pool2(tmp)
        tmp = self.conv3_1(tmp)
        feature['conv3_2'] = self.conv3_2(tmp)
        tmp = self.conv3_3(feature['conv3_2'])
        feature['conv3_4'] = self.conv3_4(tmp)
        # tmp = self.conv3_3(feature['conv3_2'])
        # tmp = self.conv3_4(tmp)
        # tmp = self.pool3(feature['conv3_4'])
        # feature['conv4_1'] = self.conv4_1(feature['pool3'])
        # feature['conv4_2'] = self.conv4_2(feature['conv4_1'])

        return feature
