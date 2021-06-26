#coding:utf-8
import cv2
import numpy as np
import sys 
import math
import os
import models_metric 
import torchvision.transforms as transforms
import torch.nn as nn 
import torch
import shutil
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path)]
def get_selected_imlist(path, txt_path, num):
    f = open(txt_path)
    cc = 0
    line = f.readline()
    L = []
    while line:
    	cc+=1
    	if cc == num:
    		break
    	name = line.split()[0]
    	L.append(path + os.sep + name + '.png')
    	line = f.readline()
    f.close()
    return L

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
a = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=np.float32)
a = a.reshape(1, 1, 3, 3) # out_c/3, in_c, w, h
a = np.repeat(a, 3, axis=0)
conv1=nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
conv1.weight.data.copy_(torch.from_numpy(a))
conv1.weight.requires_grad = False
conv1.cuda()

b = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]], dtype=np.float32)
b = b.reshape(1, 1, 3, 3)
b = np.repeat(b, 3, axis=0)
conv2=nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
conv2.weight.data.copy_(torch.from_numpy(b))
conv2.weight.requires_grad = False
conv2.cuda()

net_metric = models_metric.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, spatial=False)
net_metric = net_metric.cuda()
set_requires_grad(net_metric, requires_grad=False)


src_folder = sys.argv[1]  # The folder for demoired results by GDN
tar_folder = sys.argv[2]  # The folder for GT
txt_path = sys.argv[3]  # The txt containing image names
num = int(sys.argv[4])  # The number of candidates
front_num = int(sys.argv[5])  # the number of topK 
res_folder = src_folder + os.sep + 'd'



# res_img_path = sys.argv[1]
# target_img_path = sys.argv[2]

# res_img = cv2.imread(res_img_path)
# target_img = cv2.imread(target_img_path)
f = open("recoed_rank2_edge.txt", 'a+')
cnt = 0
D = {}
res_pre_fix = 'res'
tar_pre_fix = 'tar'
if os.path.exists('tmp1.npy'):
	A = np.load('tmp.npy')
else:
	for res_img_path in get_selected_imlist(res_folder, txt_path, num):
		cnt +=1
		if  cnt %10 ==0:
		    print(cnt)
		(filename, tempfilename) = os.path.split(res_img_path)
		(short_name, extension) = os.path.splitext(tempfilename)
		tmp = tempfilename.split('_')
		tmp2 = short_name.split('_')
		target_img_path = src_folder + os.sep + 'g' + os.sep + tar_pre_fix + '_'+tmp[1] 
		# ori_img_path = src_folder + os.sep + 'o' + os.sep + tempfilename
		res_img = cv2.imread(res_img_path)
		target_img = cv2.imread(target_img_path)
		# ori_img = cv2.imread(ori_img_path)
		gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
		gray_res = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)
		ret1, binary_map  = cv2.threshold(gray_target, 0, 255, cv2.THRESH_OTSU)

		# binary_map = binary_map / 255
		binary_map = binary_map[:, :, np.newaxis]
		binary_map = np.repeat(binary_map, 3, axis=2)


		my_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			])

		tensor_res_img = my_transform(res_img).unsqueeze_(0).cuda()
		tensor_target_img = my_transform(target_img).unsqueeze_(0).cuda()
		tensor_binary_map = my_transform(binary_map).unsqueeze_(0).cuda()
		# print(tensor_binary_map)
		tensor_binary_map = (tensor_binary_map + 1 )/2
		# print(tensor_binary_map)

		i_G_x = conv1(tensor_res_img)
		i_G_y = conv2(tensor_res_img)
		x_hat_edge = torch.tanh(torch.abs(i_G_x) + torch.abs(i_G_y))

		t_G_x = conv1(tensor_target_img)
		t_G_y = conv2(tensor_target_img)
		target_edge = torch.tanh(torch.abs(t_G_x) + torch.abs(t_G_y))

		tensor_merge_res = x_hat_edge * tensor_binary_map 
		tensor_merge_target = target_edge * tensor_binary_map
		res = net_metric(tensor_merge_res, tensor_merge_target).detach()[0][0][0][0].cpu().numpy()
		D[short_name] = res
		# break
	A = sorted(D.iteritems(), key=lambda x: x[1], reverse=True)
	np.save('tmp.npy', A)
cc = 0
for item in A:
	if cc==front_num:
		break
	f.write(str(item[0]) + ' '  + str(item[1]) + '\n')
	src_res_img_path = src_folder + os.sep + 'd' + os.sep + item[0] + '.png'
	tar_res_img_path = tar_folder + os.sep + 'd' + os.sep + item[0] + '.png'
	# tar_res_img_path = tar_folder + os.sep + 'd' + os.sep + 'src_' + '0' * (5 - len(str(cc))) + str(cc) + '.png'
	src_tar_img_path = src_folder + os.sep + 'g' + os.sep + 'tar_' + item[0].split('_')[1] + '.png'
	tar_tar_img_path = tar_folder + os.sep + 'g' + os.sep + 'tar_' + item[0].split('_')[1] + '.png'
	# tar_tar_img_path = tar_folder + os.sep + 'g' + os.sep + 'tar_' +  '0' * (5 - len(str(cc))) + str(cc) + '.png'

	shutil.copy(src_res_img_path, tar_res_img_path)
	shutil.copy(src_tar_img_path, tar_tar_img_path)
	cc+=1