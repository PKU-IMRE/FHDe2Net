import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

import sys

IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '',
]


# val_num = [0, 4991, 4979, 4967, 4993, 4995, 4963, 4978, 4994, 4731, 4971]
# test_num = [0, 9989, 8460, 9820, 9821, 9454, 9886, 3707, 4979, 4867, 4975, 4769]
def get_list(path):
	L = [os.path.join(path, f) for f in os.listdir(path)]
	return L

def is_image_file(filename):
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# def make_dataset(dir):
#   images = []
#   if not os.path.isdir(dir):
#     raise Exception('Check dataroot')
#   for root, _, fnames in sorted(os.walk(dir)):
#     for fname in fnames:
#       if is_image_file(fname):
#         path = os.path.join(dir, fname)
#         item = path
#         images.append(item)
#   return images

def make_dataset(path):
	L = [os.path.join(path, f) for f in os.listdir(path)]
	L.sort()
	return L
def make_dataset_2(dir, list_file):
  f = open(list_file)
  line = f.readline()
  L = []
  while line:
    line = line.strip()
    L.append(dir + os.sep + 'src_' + line + '.png')
    line = f.readline()
  f.close()
  return L
def make_dataset_f(dir):
  all_images = {}
  cnt = 1
  for folder_path in get_list(path):
      all_images[cnt] = get_list(folder_path)
      cnt += 1
  return all_images

def make_labels(dir, k):
    D = {}
    f = open(dir)
    line = f.readline()
    while line:
        key, v = line.split(':')
        v = v.strip('\n')
        v = int(v.split(' ')[k])
        D[key] = v
        line = f.readline()
    f.close()
    return D

def make_pos(pos_file):
    print("make_position dict")
    D = {}
    f = open(pos_file)
    line = f.readline()
    cnt = 0
    while line:
        print(cnt)
        cnt+=1
        line = line.strip()
        tmp1 = line.split(':')
        name = tmp1[0]
        tmp2 = tmp1[1].split(';')
        res = []
        for item in tmp2:
            if item == '':
                break
            tmp3 = item.split('-')
            M_x = int(tmp3[1]) - 192 # width
            M_y = int(tmp3[0]) - 192 # height
            res.append([M_x, M_y])
        D[name] = res
        line =f.readline()
    return D

def make_pos_top3(pos_file):
    print("make_position dict")
    D = {}
    f = open(pos_file)
    line = f.readline()
    cnt = 0
    while line:
        print(cnt)
        cnt+=1
        line = line.strip()
        tmp1 = line.split(':')
        name = tmp1[0]
        tmp2 = tmp1[1].split(';')
        res = []
        cnt1 = 0
        for item in tmp2:
            if item == '':
                break
            tmp3 = item.split('-')
            M_x = int(tmp3[1]) - 192 # width
            M_y = int(tmp3[0]) - 192 # height
            res.append([M_x, M_y])
            cnt1+=1
            if cnt1 == 3:
                break
        D[name] = res
        print(len(res))
        line =f.readline()
    return D
# def default_loader(path):
#   return Image.open(path).convert('RGB')

def default_loader(path):
  img = Image.open(path).convert('RGB')
  w, h = img.size
  region = img.crop((1+int(0.15*w), 1+int(0.15*h), int(0.85*w), int(0.85*h)))
  return region

class my_loader(data.Dataset):
  def __init__(self, root, transform=None, loader=default_loader, seed=None, pre="", label_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset(self.source_path)
    self.label_file = label_file
    if label_file!='':
        self.labels_0 = make_labels(label_file, 0)
        self.labels_1 = make_labels(label_file, 1)
        self.labels_2 = make_labels(label_file, 2)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform = transform
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    #filename2 = "/media/he/Seagate Expansion Drive/hebin/final_dataset/ori"
    tar_path = self.target_path + os.sep + tmp[0] + '_' + tmp[1] + "_" + tmp[2] + "_" + tmp[3] + "_" + "target" + extension
    # print(src_path)
    # print(tar_path)
    imgA = self.loader(src_path)
    imgB = self.loader(tar_path)
    if self.label_file!='':
        label_0 = self.labels_0[tempfilename]
        label_1 = self.labels_1[tempfilename]
        label_2 = self.labels_2[tempfilename]

    if self.transform is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform(imgA, imgB)

    if self.label_file != '':
        return imgA, imgB, label_0, label_1, label_2
    else:
        return imgA, imgB

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.src_imgs)

def default_loader2(path):
  img = Image.open(path).convert('RGB')
  # w, h = img.size
  # region = img.crop((1+int(0.15*w), 1+int(0.15*h), int(0.85*w), int(0.85*h)))
  return img

def default_loader3(path):

  try:
    img = Image.open(path).convert('L')
  except IOError:
    print(path)
  # w, h = img.size
  # region = img.crop((1+int(0.15*w), 1+int(0.15*h), int(0.85*w), int(0.85*h)))
  return img

def default_loader4(path):

  try:
    img = Image.open(path).split()[1]
  except IOError:
    print(path)
  # w, h = img.size
  # region = img.crop((1+int(0.15*w), 1+int(0.15*h), int(0.85*w), int(0.85*h)))
  return img

class my_loader2(data.Dataset):
  def __init__(self, root, transform=None, loader=default_loader2, seed=None, pre="", label_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset(self.source_path)
    self.label_file = label_file
    if label_file!='':
        self.labels_0 = make_labels(label_file, 0)
        self.labels_1 = make_labels(label_file, 1)
        self.labels_2 = make_labels(label_file, 2)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform = transform
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    #filename2 = "/media/he/Seagate Expansion Drive/hebin/final_dataset/ori"
    # tar_path = self.target_path + os.sep + tmp[0] + '_' + tmp[1] + "_" + tmp[2] + "_" + tmp[3] + "_" + "target" + extension
    tar_path =src_path
    # print(src_path)
    # print(tar_path)
    imgA = self.loader(src_path)
    imgB = self.loader(tar_path)
    if self.label_file!='':
        label_0 = self.labels_0[tempfilename]
        label_1 = self.labels_1[tempfilename]
        label_2 = self.labels_2[tempfilename]

    if self.transform is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform(imgA, imgB)

    if self.label_file != '':
        return imgA, imgB, label_0, label_1, label_2
    else:
        return imgA, imgB

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.src_imgs)

class my_loader_predictor(data.Dataset):
  def __init__(self, root, transform=None, loader=default_loader2, seed=None, pre="", label_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.moire_path = root+os.sep+pre+"moire"
    src_imgs = make_dataset(self.source_path)


    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    self.root = root
    self.src_imgs = src_imgs
    self.transform = transform
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    moire_path =self.moire_path +os.sep+ "mo_" + tmp[1] + extension

    imgA = self.loader(src_path)
    imgB = self.loader(moire_path)

    if self.transform is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform(imgA, imgB)


    return imgA, imgB

  def __len__(self):
    return len(self.src_imgs)


class my_loader_LSN_guide(data.Dataset):
  def __init__(self, root, transform1=None, transform2=None, transform3=None, transform4=None, transform5=None, transform6=None,loader=default_loader2, seed=None, pre="", label_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset(self.source_path)
    self.label_file = label_file
    if label_file!='':
        self.pos_dict = make_pos(label_file)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform1 = transform1
    self.transform2 = transform2
    self.transform3 = transform3
    self.transform4 = transform4
    self.transform5 = transform5
    self.transform6 = transform6
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    tar_path = self.target_path + os.sep + 'tar' + '_' + tmp[1] + '.png'
    imgA = self.loader(src_path)
    imgB = self.loader(tar_path)
    pos_list = self.pos_dict[short_name]

    if self.transform1 is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform1(imgA, imgB)
      imgA, imgB, ox1, oy1 = self.transform2(imgA, imgB)
      imgA, imgB, flag = self.transform3(imgA, imgB)
      if np.random.randint(0,10) % 2 ==0:
          x1 = -1
          y1 = -1
      else:
          cnt1 = 0
          while(True):
              cnt1+=1
              if cnt1>30:
                  x1 = -1
                  y1 = -1
                  print("no matched case!!!!!!!")
                  break
              index = np.random.randint(0, 7)
              ori_x = pos_list[index][0] + np.random.randint(0,50)
              ori_y = pos_list[index][1] + np.random.randint(0,50)
              # print("o:" , ox1, oy1)
              # print("ori", ori_x, ori_y)
              if ori_x >= ox1 and ori_y>=oy1 and ori_x -ox1 + 384 <1024 and ori_y - oy1 + 384 < 1024:
                  x1 = ori_x - ox1
                  y1 = ori_y - oy1
                  if flag == True:
                      x1 = 1024 - x1 -384
                  break
      # print(x1,y1)
      imgC = self.transform6(imgA)
      imgA, imgB = self.transform4(x1, y1, imgA, imgB)
      imgA, imgB = self.transform5(imgA, imgB)


    return imgA, imgB, imgC

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.src_imgs)



class my_loader_LSN_pure_guide(data.Dataset):
  def __init__(self, root, transform1=None, transform2=None, transform3=None, transform4=None, transform5=None, transform6=None,loader=default_loader2, seed=None, pre="", label_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset(self.source_path)
    self.label_file = label_file
    if label_file!='':
        self.pos_dict = make_pos(label_file)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform1 = transform1
    self.transform2 = transform2
    self.transform3 = transform3
    self.transform4 = transform4
    self.transform5 = transform5
    self.transform6 = transform6
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    tar_path = self.target_path + os.sep + 'tar' + '_' + tmp[1] + '.png'
    imgA = self.loader(src_path)
    imgB = self.loader(tar_path)
    pos_list = self.pos_dict[short_name]

    if self.transform1 is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform1(imgA, imgB)
      imgA, imgB, ox1, oy1 = self.transform2(imgA, imgB)
      imgA, imgB, flag = self.transform3(imgA, imgB)
      # if np.random.randint(0,10) % 2 ==0:
      #     x1 = -1
      #     y1 = -1
      # else:
      cnt1 = 0
      while(True):
          cnt1+=1
          if cnt1>30:
              x1 = -1
              y1 = -1
              print("no matched case!!!!!!!")
              break
          index = np.random.randint(0, 7)
          ori_x = pos_list[index][0] + np.random.randint(0,30)
          ori_y = pos_list[index][1] + np.random.randint(0,30)
          # print("o:" , ox1, oy1)
          # print("ori", ori_x, ori_y)
          if ori_x >= ox1 and ori_y>=oy1 and ori_x -ox1 + 384 <1024 and ori_y - oy1 + 384 < 1024:
              x1 = ori_x - ox1
              y1 = ori_y - oy1
              if flag == True:
                  x1 = 1024 - x1 - 384
              break
      # print(x1,y1)
      imgC = self.transform6(imgA)
      imgA, imgB = self.transform4(x1, y1, imgA, imgB)
      imgA, imgB = self.transform5(imgA, imgB)


    return imgA, imgB, imgC

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.src_imgs)


class my_loader_LSN_real_pure_guide(data.Dataset):
  def __init__(self, root, transform1=None, transform2=None, transform3=None, transform4=None, transform5=None, transform6=None,loader=default_loader2, seed=None, pre="", label_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset(self.source_path)
    self.label_file = label_file
    if label_file!='':
        self.pos_dict = make_pos(label_file)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform1 = transform1
    self.transform2 = transform2
    self.transform3 = transform3
    self.transform4 = transform4
    self.transform5 = transform5
    self.transform6 = transform6
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    tar_path = self.target_path + os.sep + 'tar' + '_' + tmp[1] + '.png'
    imgA_ = self.loader(src_path)
    imgB_ = self.loader(tar_path)
    pos_list = self.pos_dict[short_name]

    if self.transform1 is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform1(imgA_, imgB_)

      while(True):
          imgA, imgB, ox1, oy1 = self.transform2(imgA_, imgB_)
          imgA, imgB, flag = self.transform3(imgA, imgB)
          k = False
          # if np.random.randint(0,10) % 2 ==0:
          #     x1 = -1
          #     y1 = -1
          # else:
          cnt1 = 0
          while(True):
              cnt1+=1
              if cnt1>20:
                  x1 = -1
                  y1 = -1
                  print("no matched case!!!!!!!")
                  break
              index = np.random.randint(0, 7)
              ori_x = pos_list[index][0] + np.random.randint(0,30)
              ori_y = pos_list[index][1] + np.random.randint(0,30)
              # print("o:" , ox1, oy1)
              # print("ori", ori_x, ori_y)
              if ori_x >= ox1 and ori_y>=oy1 and ori_x -ox1 + 384 <1024 and ori_y - oy1 + 384 < 1024:
                  x1 = ori_x - ox1
                  y1 = ori_y - oy1
                  if flag == True:
                      x1 = 1024 - x1 - 384
                  k = True
                  break
          if k==True:
              break

      # print(x1,y1)
      imgC = self.transform6(imgA)
      imgA, imgB = self.transform4(x1, y1, imgA, imgB)
      imgA, imgB = self.transform5(imgA, imgB)


    return imgA, imgB, imgC

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.src_imgs)


class my_loader_LSN_real_pure_guide_P_dct(data.Dataset):
  def __init__(self, root, transform1=None, transform2=None, transform3=None, transform4=None, transform5=None, transform6=None,loader=default_loader3, seed=None, pre="", label_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset(self.source_path)
    self.label_file = label_file
    if label_file!='':
        self.pos_dict = make_pos(label_file)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform1 = transform1
    self.transform2 = transform2
    self.transform3 = transform3
    self.transform4 = transform4
    self.transform5 = transform5
    self.transform6 = transform6
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    tar_path = self.target_path + os.sep + 'tar' + '_' + tmp[1] + '.png'
    imgA_ = self.loader(src_path)
    imgB_ = self.loader(tar_path)
    # print(short_name)
    pos_list = self.pos_dict[short_name]

    if self.transform1 is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform1(imgA_, imgB_)

      while(True):
          imgA, imgB, ox1, oy1 = self.transform2(imgA_, imgB_)
          imgA, imgB, flag = self.transform3(imgA, imgB)
          k = False
          # if np.random.randint(0,10) % 2 ==0:
          #     x1 = -1
          #     y1 = -1
          # else:
          cnt1 = 0
          while(True):
              cnt1+=1
              if cnt1>20:
                  x1 = -1
                  y1 = -1
                  print("no matched case!!!!!!!")
                  break
              index = np.random.randint(0, 7)
              ori_x = pos_list[index][0] + np.random.randint(0,30)
              ori_y = pos_list[index][1] + np.random.randint(0,30)
              # print("o:" , ox1, oy1)
              # print("ori", ori_x, ori_y)
              if ori_x >= ox1 and ori_y>=oy1 and ori_x -ox1 + 384 <1024 and ori_y - oy1 + 384 < 1024:
                  x1 = ori_x - ox1
                  y1 = ori_y - oy1
                  if flag == True:
                      x1 = 1024 - x1 - 384
                  k = True
                  break
          if k==True:
              break

      # print(x1,y1)
      imgC = self.transform6(imgA)
      imgA, imgB = self.transform4(x1, y1, imgA, imgB)
      imgA, imgB = self.transform5(imgA, imgB)


    return imgA, imgB, imgC

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.src_imgs)


class my_loader_LSN_rand_crop_P_dct(data.Dataset):
  def __init__(self, root, transform=None, loader=default_loader3, seed=None, pre="", label_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset(self.source_path)
    # self.label_file = label_file
    # if label_file!='':
    #     self.labels_0 = make_labels(label_file, 0)
    #     self.labels_1 = make_labels(label_file, 1)
    #     self.labels_2 = make_labels(label_file, 2)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform = transform
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    tar_path = self.target_path + os.sep + 'tar' + '_' + tmp[1] + '.png'
    imgA = self.loader(src_path)
    imgB = self.loader(tar_path)
    # if self.label_file!='':
    #     label_0 = self.labels_0[tempfilename]
    #     label_1 = self.labels_1[tempfilename]
    #     label_2 = self.labels_2[tempfilename]

    if self.transform is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform(imgA, imgB)
    #
    # if self.label_file != '':
    #     return imgA, imgB, label_0, label_1, label_2
    # else:
    return imgA, imgB

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.src_imgs)

class my_loader_LSN_rand_crop_P_dct_2(data.Dataset):
  def __init__(self, root, transform=None, loader=default_loader3, seed=None, pre="", label_file='', list_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset_2(self.source_path, list_file)
    # self.label_file = label_file
    # if label_file!='':
    #     self.labels_0 = make_labels(label_file, 0)
    #     self.labels_1 = make_labels(label_file, 1)
    #     self.labels_2 = make_labels(label_file, 2)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform = transform
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    tar_path = self.target_path + os.sep + 'tar' + '_' + tmp[1] + '.png'
    imgA = self.loader(src_path)
    imgB = self.loader(tar_path)
    # if self.label_file!='':
    #     label_0 = self.labels_0[tempfilename]
    #     label_1 = self.labels_1[tempfilename]
    #     label_2 = self.labels_2[tempfilename]

    if self.transform is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform(imgA, imgB)
    #
    # if self.label_file != '':
    #     return imgA, imgB, label_0, label_1, label_2
    # else:
    return imgA, imgB

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.src_imgs)

class my_loader_LSN_f2_rand2(data.Dataset):
  def __init__(self, root, transform1=None, transform2=None, transform3=None, transform4=None, transform5=None, transform6=None,loader=default_loader2, seed=None, pre="", label_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset(self.source_path)
    self.label_file = label_file
    if label_file!='':
        self.pos_dict = make_pos(label_file)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform1 = transform1
    self.transform2 = transform2
    self.transform3 = transform3
    self.transform4 = transform4
    self.transform5 = transform5
    self.transform6 = transform6
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    tar_path = self.target_path + os.sep + 'tar' + '_' + tmp[1] + '.png'
    imgA_ = self.loader(src_path)
    imgB_ = self.loader(tar_path)
    pos_list = self.pos_dict[short_name]

    if self.transform1 is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform1(imgA_, imgB_)


      imgA, imgB, ox1, oy1 = self.transform2(imgA_, imgB_)
      imgA, imgB, flag = self.transform3(imgA, imgB)
      k = False
      # if np.random.randint(0,10) % 2 ==0:
      #     x1 = -1
      #     y1 = -1
      # else:
      x1 = np.random.randint(0, 1024 - 384 -1)
      y1 = np.random.randint(0, 1024 - 384 - 1)
      # print(x1,y1)
      imgC = self.transform6(imgA)
      imgD = self.transform6(imgB)
      imgA, imgB = self.transform4(x1, y1, imgA, imgB)
      imgA, imgB = self.transform5(imgA, imgB)


    return imgA, imgB, imgC, imgD, x1, y1

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.src_imgs)

class my_loader_LSN_real_pure_guide_P_dct_top3(data.Dataset):
  def __init__(self, root, transform1=None, transform2=None, transform3=None, transform4=None, transform5=None, transform6=None,loader=default_loader3, seed=None, pre="", label_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset(self.source_path)
    self.label_file = label_file
    if label_file!='':
        self.pos_dict = make_pos_top3(label_file)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform1 = transform1
    self.transform2 = transform2
    self.transform3 = transform3
    self.transform4 = transform4
    self.transform5 = transform5
    self.transform6 = transform6
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    tar_path = self.target_path + os.sep + 'tar' + '_' + tmp[1] + '.png'
    imgA_ = self.loader(src_path)
    imgB_ = self.loader(tar_path)
    # print(short_name)
    pos_list = self.pos_dict[short_name]

    if self.transform1 is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform1(imgA_, imgB_)

      while(True):
          imgA, imgB, ox1, oy1 = self.transform2(imgA_, imgB_)
          imgA, imgB, flag = self.transform3(imgA, imgB)
          k = False
          # if np.random.randint(0,10) % 2 ==0:
          #     x1 = -1
          #     y1 = -1
          # else:
          cnt1 = 0
          while(True):
              cnt1+=1
              if cnt1>20:
                  x1 = -1
                  y1 = -1
                  print("no matched case!!!!!!!")
                  break
              index = np.random.randint(0, 3)
              ori_x = pos_list[index][0] + np.random.randint(0,30)
              ori_y = pos_list[index][1] + np.random.randint(0,30)
              # print("o:" , ox1, oy1)
              # print("ori", ori_x, ori_y)
              if ori_x >= ox1 and ori_y>=oy1 and ori_x -ox1 + 384 <1024 and ori_y - oy1 + 384 < 1024:
                  x1 = ori_x - ox1
                  y1 = ori_y - oy1
                  if flag == True:
                      x1 = 1024 - x1 - 384
                  k = True
                  break
          if k==True:
              break

      # print(x1,y1)
      imgC = self.transform6(imgA)
      imgA, imgB = self.transform4(x1, y1, imgA, imgB)
      imgA, imgB = self.transform5(imgA, imgB)


    return imgA, imgB, imgC

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.src_imgs)


class my_loader_LSN_real_pure_guide_P_dct_green(data.Dataset):
  def __init__(self, root, transform1=None, transform2=None, transform3=None, transform4=None, transform5=None, transform6=None,loader=default_loader4, seed=None, pre="", label_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset(self.source_path)
    self.label_file = label_file
    if label_file!='':
        self.pos_dict = make_pos(label_file)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform1 = transform1
    self.transform2 = transform2
    self.transform3 = transform3
    self.transform4 = transform4
    self.transform5 = transform5
    self.transform6 = transform6
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    tar_path = self.target_path + os.sep + 'tar' + '_' + tmp[1] + '.png'
    imgA_ = self.loader(src_path)
    imgB_ = self.loader(tar_path)
    pos_list = self.pos_dict[short_name]

    if self.transform1 is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform1(imgA_, imgB_)

      while(True):
          imgA, imgB, ox1, oy1 = self.transform2(imgA_, imgB_)
          imgA, imgB, flag = self.transform3(imgA, imgB)
          k = False
          # if np.random.randint(0,10) % 2 ==0:
          #     x1 = -1
          #     y1 = -1
          # else:
          cnt1 = 0
          while(True):
              cnt1+=1
              if cnt1>20:
                  x1 = -1
                  y1 = -1
                  print("no matched case!!!!!!!")
                  break
              index = np.random.randint(0, 7)
              ori_x = pos_list[index][0] + np.random.randint(0,30)
              ori_y = pos_list[index][1] + np.random.randint(0,30)
              # print("o:" , ox1, oy1)
              # print("ori", ori_x, ori_y)
              if ori_x >= ox1 and ori_y>=oy1 and ori_x -ox1 + 384 <1024 and ori_y - oy1 + 384 < 1024:
                  x1 = ori_x - ox1
                  y1 = ori_y - oy1
                  if flag == True:
                      x1 = 1024 - x1 - 384
                  k = True
                  break
          if k==True:
              break

      # print(x1,y1)
      imgC = self.transform6(imgA)
      imgA, imgB = self.transform4(x1, y1, imgA, imgB)
      imgA, imgB = self.transform5(imgA, imgB)


    return imgA, imgB, imgC

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.src_imgs)

class my_loader_LSN_f1(data.Dataset):
  def __init__(self, root, transform1=None, transform2=None, transform3=None, transform4=None, transform5=None, transform6=None,loader=default_loader2, seed=None, pre="", label_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset(self.source_path)
    self.label_file = label_file
    if label_file!='':
        self.pos_dict = make_pos(label_file)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform1 = transform1
    self.transform2 = transform2
    self.transform3 = transform3
    self.transform4 = transform4
    self.transform5 = transform5
    self.transform6 = transform6
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    tar_path = self.target_path + os.sep + 'tar' + '_' + tmp[1] + '.png'
    imgA_ = self.loader(src_path)
    imgB_ = self.loader(tar_path)
    pos_list = self.pos_dict[short_name]

    if self.transform1 is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform1(imgA_, imgB_)

      while(True):
          imgA, imgB, ox1, oy1 = self.transform2(imgA_, imgB_)
          imgA, imgB, flag = self.transform3(imgA, imgB)
          k = False
          # if np.random.randint(0,10) % 2 ==0:
          #     x1 = -1
          #     y1 = -1
          # else:
          cnt1 = 0
          while(True):
              cnt1+=1
              if cnt1>20:
                  x1 = -1
                  y1 = -1
                  print("no matched case!!!!!!!")
                  break
              index = np.random.randint(0, 7)
              ori_x = pos_list[index][0] + np.random.randint(0,30)
              ori_y = pos_list[index][1] + np.random.randint(0,30)
              # print("o:" , ox1, oy1)
              # print("ori", ori_x, ori_y)
              if ori_x >= ox1 and ori_y>=oy1 and ori_x -ox1 + 384 <1024 and ori_y - oy1 + 384 < 1024:
                  x1 = ori_x - ox1
                  y1 = ori_y - oy1
                  if flag == True:
                      x1 = 1024 - x1 - 384
                  k = True
                  break
          if k==True:
              break

      # print(x1,y1)
      imgC = self.transform6(imgA)
      imgA, imgB = self.transform4(x1, y1, imgA, imgB)
      imgA, imgB = self.transform5(imgA, imgB)


    return imgA, imgB, imgC, x1, y1

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.src_imgs)


class my_loader_LSN_f1_top3(data.Dataset):
  def __init__(self, root, transform1=None, transform2=None, transform3=None, transform4=None, transform5=None, transform6=None,loader=default_loader2, seed=None, pre="", label_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset(self.source_path)
    self.label_file = label_file
    if label_file!='':
        self.pos_dict = make_pos_top3(label_file)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform1 = transform1
    self.transform2 = transform2
    self.transform3 = transform3
    self.transform4 = transform4
    self.transform5 = transform5
    self.transform6 = transform6
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    tar_path = self.target_path + os.sep + 'tar' + '_' + tmp[1] + '.png'
    imgA_ = self.loader(src_path)
    imgB_ = self.loader(tar_path)
    pos_list = self.pos_dict[short_name]

    if self.transform1 is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform1(imgA_, imgB_)

      while(True):
          imgA, imgB, ox1, oy1 = self.transform2(imgA_, imgB_)
          imgA, imgB, flag = self.transform3(imgA, imgB)
          k = False
          # if np.random.randint(0,10) % 2 ==0:
          #     x1 = -1
          #     y1 = -1
          # else:
          cnt1 = 0
          while(True):
              cnt1+=1
              if cnt1>20:
                  x1 = -1
                  y1 = -1
                  print("no matched case!!!!!!!")
                  break
              index = np.random.randint(0, 3)
              ori_x = pos_list[index][0] + np.random.randint(0,30)
              ori_y = pos_list[index][1] + np.random.randint(0,30)
              # print("o:" , ox1, oy1)
              # print("ori", ori_x, ori_y)
              if ori_x >= ox1 and ori_y>=oy1 and ori_x -ox1 + 384 <1024 and ori_y - oy1 + 384 < 1024:
                  x1 = ori_x - ox1
                  y1 = ori_y - oy1
                  if flag == True:
                      x1 = 1024 - x1 - 384
                  k = True
                  break
          if k==True:
              break

      # print(x1,y1)
      imgC = self.transform6(imgA)
      imgA, imgB = self.transform4(x1, y1, imgA, imgB)
      imgA, imgB = self.transform5(imgA, imgB)


    return imgA, imgB, imgC, x1, y1

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.src_imgs)

class my_loader_LSN_f2_rand(data.Dataset):
  def __init__(self, root, transform1=None, transform2=None, transform3=None, transform4=None, transform5=None, transform6=None,loader=default_loader2, seed=None, pre="", label_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset(self.source_path)
    self.label_file = label_file
    if label_file!='':
        self.pos_dict = make_pos(label_file)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform1 = transform1
    self.transform2 = transform2
    self.transform3 = transform3
    self.transform4 = transform4
    self.transform5 = transform5
    self.transform6 = transform6
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    tar_path = self.target_path + os.sep + 'tar' + '_' + tmp[1] + '.png'
    imgA_ = self.loader(src_path)
    imgB_ = self.loader(tar_path)
    pos_list = self.pos_dict[short_name]

    if self.transform1 is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform1(imgA_, imgB_)


      imgA, imgB, ox1, oy1 = self.transform2(imgA_, imgB_)
      imgA, imgB, flag = self.transform3(imgA, imgB)
      k = False
      # if np.random.randint(0,10) % 2 ==0:
      #     x1 = -1
      #     y1 = -1
      # else:
      x1 = np.random.randint(0, 1024 - 384 -1)
      y1 = np.random.randint(0, 1024 - 384 - 1)
      # print(x1,y1)
      imgC = self.transform6(imgA)
      imgA, imgB = self.transform4(x1, y1, imgA, imgB)
      imgA, imgB = self.transform5(imgA, imgB)


    return imgA, imgB, imgC, x1, y1

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.src_imgs)


class my_loader_rand_crop(data.Dataset):
  def __init__(self, root, transform=None, loader=default_loader2, seed=None, pre="", label_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset(self.source_path)
    self.label_file = label_file
    if label_file!='':
        self.labels_0 = make_labels(label_file, 0)
        self.labels_1 = make_labels(label_file, 1)
        self.labels_2 = make_labels(label_file, 2)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform = transform
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    tar_path = self.target_path + os.sep + 'tar' + '_' + tmp[1] + '.png'
    imgA = self.loader(src_path)
    imgB = self.loader(tar_path)
    if self.label_file!='':
        label_0 = self.labels_0[tempfilename]
        label_1 = self.labels_1[tempfilename]
        label_2 = self.labels_2[tempfilename]

    if self.transform is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform(imgA, imgB)

    if self.label_file != '':
        return imgA, imgB, label_0, label_1, label_2
    else:
        return imgA, imgB

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.src_imgs)


class my_loader_fs_2(data.Dataset):
  def __init__(self, root, transform=None, loader=default_loader2, seed=None, pre="", label_file='', list_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset_2(self.source_path, list_file)
    self.label_file = label_file
    if label_file!='':
        self.labels_0 = make_labels(label_file, 0)
        self.labels_1 = make_labels(label_file, 1)
        self.labels_2 = make_labels(label_file, 2)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform = transform
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    # tar_path = self.target_path + os.sep + 'tar' + '_' + tmp[1] + '_' + tmp[2] + '.png'
    tar_path = self.target_path + os.sep + 'tar' + '_' + tmp[1] + '.png'
    imgA = self.loader(src_path)
    imgB = self.loader(tar_path)
    if self.label_file!='':
        label_0 = self.labels_0[tempfilename]
        label_1 = self.labels_1[tempfilename]
        label_2 = self.labels_2[tempfilename]

    if self.transform is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform(imgA, imgB)

    if self.label_file != '':
        return imgA, imgB, label_0, label_1, label_2
    else:
        return imgA, imgB, tempfilename

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.src_imgs)

class my_loader_fs(data.Dataset):
  def __init__(self, root, transform=None, loader=default_loader2, seed=None, pre="", label_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset(self.source_path)
    self.label_file = label_file
    if label_file!='':
        self.labels_0 = make_labels(label_file, 0)
        self.labels_1 = make_labels(label_file, 1)
        self.labels_2 = make_labels(label_file, 2)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform = transform
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    # tar_path = self.target_path + os.sep + 'tar' + '_' + tmp[1] + '_' + tmp[2] + '.png'
    tar_path = self.target_path + os.sep + 'tar' + '_' + tmp[1] + '.png'
    imgA = self.loader(src_path)
    imgB = self.loader(tar_path)
    if self.label_file!='':
        label_0 = self.labels_0[tempfilename]
        label_1 = self.labels_1[tempfilename]
        label_2 = self.labels_2[tempfilename]

    if self.transform is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform(imgA, imgB)

    if self.label_file != '':
        return imgA, imgB, label_0, label_1, label_2
    else:
        return imgA, imgB, tempfilename

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.src_imgs)

class my_loader_LSN_gray(data.Dataset):
  def __init__(self, root, transform1=None, transform2=None, transform3=None, loader=default_loader3, seed=None, pre="", label_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset(self.source_path)
    self.label_file = label_file
    if label_file!='':
        self.labels_0 = make_labels(label_file, 0)
        self.labels_1 = make_labels(label_file, 1)
        self.labels_2 = make_labels(label_file, 2)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform1 = transform1
    self.transform2 = transform2
    self.transform3 = transform3
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    tar_path = self.target_path + os.sep + 'tar' + '_' + tmp[1] + '.png'
    imgA = self.loader(src_path)
    imgB = self.loader(tar_path)
    if self.label_file!='':
        label_0 = self.labels_0[tempfilename]
        label_1 = self.labels_1[tempfilename]
        label_2 = self.labels_2[tempfilename]

    if self.transform1 is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform1(imgA, imgB)
      imgC = self.transform3(imgA)
      imgA, imgB = self.transform2(imgA, imgB)

    if self.label_file != '':
        return imgA, imgB, label_0, label_1, label_2
    else:
        return imgA, imgB, imgC

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.src_imgs)

class my_loader_DCTN(data.Dataset):
  def __init__(self, root, transform1=None, transform2=None, transform3=None, loader=default_loader2, seed=None, pre="", label_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset(self.source_path)
    self.label_file = label_file
    if label_file!='':
        self.labels_0 = make_labels(label_file, 0)
        self.labels_1 = make_labels(label_file, 1)
        self.labels_2 = make_labels(label_file, 2)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform1 = transform1
    self.transform2 = transform2
    self.transform3 = transform3
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    tar_path = self.target_path + os.sep + 'tar' + '_' + tmp[1] + '.png'
    imgA = self.loader(src_path)
    imgB = self.loader(tar_path)
    if self.label_file!='':
        label_0 = self.labels_0[tempfilename]
        label_1 = self.labels_1[tempfilename]
        label_2 = self.labels_2[tempfilename]

    if self.transform1 is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform1(imgA, imgB)
      imgC = self.transform3(imgA)
      imgA, imgB = self.transform2(imgA, imgB)

    if self.label_file != '':
        return imgA, imgB, label_0, label_1, label_2
    else:
        return imgA, imgB, imgC

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.src_imgs)


class my_loader_LSN_green(data.Dataset):
  def __init__(self, root, transform1=None, transform2=None, transform3=None, loader=default_loader4, seed=None, pre="", label_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset(self.source_path)
    self.label_file = label_file
    if label_file!='':
        self.labels_0 = make_labels(label_file, 0)
        self.labels_1 = make_labels(label_file, 1)
        self.labels_2 = make_labels(label_file, 2)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform1 = transform1
    self.transform2 = transform2
    self.transform3 = transform3
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    tar_path = self.target_path + os.sep + 'tar' + '_' + tmp[1] + '.png'
    imgA = self.loader(src_path)
    imgB = self.loader(tar_path)
    if self.label_file!='':
        label_0 = self.labels_0[tempfilename]
        label_1 = self.labels_1[tempfilename]
        label_2 = self.labels_2[tempfilename]

    if self.transform1 is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform1(imgA, imgB)
      imgC = self.transform3(imgA)
      imgA, imgB = self.transform2(imgA, imgB)

    if self.label_file != '':
        return imgA, imgB, label_0, label_1, label_2
    else:
        return imgA, imgB, imgC

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.src_imgs)

class my_loader_LSN(data.Dataset):
  def __init__(self, root, transform1=None, transform2=None, transform3=None, loader=default_loader2, seed=None, pre="", label_file=''):
    self.source_path = root+os.sep+pre+"source"
    self.target_path = root+os.sep+pre+"target"
    src_imgs = make_dataset(self.source_path)
    print(src_imgs)
    self.label_file = label_file
    if label_file!='':
        self.labels_0 = make_labels(label_file, 0)
        self.labels_1 = make_labels(label_file, 1)
        self.labels_2 = make_labels(label_file, 2)
    if len(src_imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root

    self.src_imgs = src_imgs
    self.transform1 = transform1
    self.transform2 = transform2
    self.transform3 = transform3
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):

    src_path = self.src_imgs[index]
    (filename, tempfilename) = os.path.split(src_path)
    (short_name, extension) = os.path.splitext(tempfilename)
    tmp = short_name.split('_')
    tar_path = self.target_path + os.sep + 'tar' + '_' + tmp[1] + '.png'
    imgA = self.loader(src_path)
    imgB = self.loader(tar_path)
    if self.label_file!='':
        label_0 = self.labels_0[tempfilename]
        label_1 = self.labels_1[tempfilename]
        label_2 = self.labels_2[tempfilename]

    if self.transform1 is not None:
      # NOTE preprocessing for each pair of images
      imgA, imgB = self.transform1(imgA, imgB)
      imgC = self.transform3(imgA)
      imgA, imgB = self.transform2(imgA, imgB)

    if self.label_file != '':
        return imgA, imgB, label_0, label_1, label_2
    else:
        return imgA, imgB, imgC

  def __len__(self):
    # return 679
    # print(len(self.imgs))
    return len(self.src_imgs)
# val_part001:00004991
# val_part002:00004979
# val_part003:00004967
# val_part004:00004993
# val_part005:00004995
# val_part006:00004963
# val_part007:00004978
# val_part008:00004994
# val_part009:00004731
# val_part010:00004971

# test_part001:00009989
# test_part002:00008460
# test_part003:00009820
# test_part004:00009821
# test_part005:00009454
# test_part006:00009886
# test_part007:00003707
# test_part008:00004979
# test_part009:00004867
# test_part011:00004975
# test_part010:00004769


def main():
    my_root = "/media/he/FE2CA0442C9FF5BD/test/"
    print(my_root)
    loader = my_loader(my_root, pre="thin_")
    print(len(loader))
    my_iterator = enumerate(loader, 0)
    # for i in range(len(loader)):
    #   _,l = my_iterator.next()
    #   print(l[0],l[1])

    # my_iterator = enumerate(loader, 0)
    # for i in range(len(loader)):
    #   print(my_iterator.next())
    # for i, data in enumerate(loader, 0):
    #     print(i)


if __name__ == '__main__':
    main()

