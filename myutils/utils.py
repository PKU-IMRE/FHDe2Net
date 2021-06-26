import os

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
# from torch.utils.serialization import load_lua
import torchfile
from myutils.vgg16 import Vgg16

from torch.optim import lr_scheduler
def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
	img = Image.open(filename).convert('RGB')
	if size is not None:
		if keep_asp:
			size2 = int(size * 1.0 / img.size[0] * img.size[1])
			img = img.resize((size, size2), Image.ANTIALIAS)
		else:
			img = img.resize((size, size), Image.ANTIALIAS)

	elif scale is not None:
		img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
	img = np.array(img).transpose(2, 0, 1)
	img = torch.from_numpy(img).float()
	return img


def tensor_save_rgbimage(tensor, filename, cuda=False):
	if cuda:
		img = tensor.clone().cpu().clamp(0, 255).numpy()
	else:
		img = tensor.clone().clamp(0, 255).numpy()
	img = img.transpose(1, 2, 0).astype('uint8')
	img = Image.fromarray(img)
	img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
	(b, g, r) = torch.chunk(tensor, 3)
	tensor = torch.cat((r, g, b))
	tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
	(b, ch, h, w) = y.size()
	features = y.view(b, ch, w * h)
	features_t = features.transpose(1, 2)
	gram = features.bmm(features_t) / (ch * h * w)
	return gram


def subtract_imagenet_mean_batch(batch):
	"""Subtract ImageNet mean pixel-wise from a BGR image."""
	tensortype = type(batch.data)
	mean = tensortype(batch.data.size())
	mean[:, 0, :, :] = 103.939
	mean[:, 1, :, :] = 116.779
	mean[:, 2, :, :] = 123.680
	return batch - Variable(mean)


def add_imagenet_mean_batch(batch):
	"""Add ImageNet mean pixel-wise from a BGR image."""
	tensortype = type(batch.data)
	mean = tensortype(batch.data.size())
	mean[:, 0, :, :] = 103.939
	mean[:, 1, :, :] = 116.779
	mean[:, 2, :, :] = 123.680
	return batch + Variable(mean)

def imagenet_clamp_batch(batch, low, high):
	batch[:,0,:,:].data.clamp_(low-103.939, high-103.939)
	batch[:,1,:,:].data.clamp_(low-116.779, high-116.779)
	batch[:,2,:,:].data.clamp_(low-123.680, high-123.680)


def preprocess_batch(batch):
	batch = batch.transpose(0, 1)
	(r, g, b) = torch.chunk(batch, 3)
	batch = torch.cat((b, g, r))
	batch = batch.transpose(0, 1)
	return batch


def init_vgg16(model_folder):
	"""load the vgg16 model feature"""
	if not os.path.exists(os.path.join(model_folder, 'vgg16.weight')):
		if not os.path.exists(os.path.join(model_folder, 'vgg16.t7')):
			os.system(
				'wget http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7 -O ' + os.path.join(model_folder, 'vgg16.t7'))
		vgglua = torchfile.load(os.path.join(model_folder, 'vgg16.t7'))
		vgg = Vgg16()
		for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
			dst.data[:] = src
		torch.save(vgg.state_dict(), os.path.join(model_folder, 'vgg16.weight'))


def set_requires_grad(nets, requires_grad=False):
	"""Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
	if not isinstance(nets, list):
		nets = [nets]
	for net in nets:
		if net is not None:
			for param in net.parameters():
				param.requires_grad = requires_grad

def get_scheduler(optimizer, opt):
	if opt.lr_policy == 'lambda':
		def lambda_rule(epoch):
			print(epoch)
			print(opt.epoch_count)
			print(opt.niter)

			lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
			print(lr_l)
			print(50*'-')
			return lr_l
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif opt.lr_policy == 'step':
		scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
	elif opt.lr_policy == 'plateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
	else:
		return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
	return scheduler

def my_tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    # print(50*'-')
    # print(input_image.shape)
    # print(image_tensor.shape)
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)