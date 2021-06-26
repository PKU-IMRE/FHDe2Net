import torch
import os 
import sys


def create_exp_dir(exp):
  try:
    os.makedirs(exp)
    print('Creating exp dir: %s' % exp)
  except OSError:
    pass
  return True


def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)


def getLoader(datasetName, dataroot, originalSize_h, originalSize_w, imageSize_h, imageSize_w, batchSize=64, workers=4,
              mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), split='train', shuffle=True, seed=None, pre="", label_file="", list_file=""):


  if datasetName == 'my_loader':
    from datasets.my_loader import my_loader_LRN as commonDataset
    import transforms.pix2pix as transforms

  elif datasetName == 'my_loader_fs':
    # from datasets.pix2pix import pix2pix as commonDataset
    # import transforms.pix2pix as transforms
    from datasets.my_loader import my_loader_fs as commonDataset
    import transforms.pix2pix as transforms
  elif datasetName == 'my_loader_LRN_f2_rand3':
    # from datasets.pix2pix import pix2pix as commonDataset
    # import transforms.pix2pix as transforms
    from datasets.my_loader import my_loader_LRN_f2_rand3 as commonDataset
    import transforms.pix2pix as transforms
  elif datasetName == 'my_loader_LRN_f2_rand2':
    # from datasets.pix2pix import pix2pix as commonDataset
    # import transforms.pix2pix as transforms
    from datasets.my_loader import my_loader_LRN_f2_rand2 as commonDataset
    import transforms.pix2pix as transforms
  if split == 'test':
    dataset = commonDataset(root=dataroot,
                            transform1=transforms.Compose([
                              transforms.Scale(originalSize_h, originalSize_w),
                              transforms.CenterCrop(imageSize_h, imageSize_w),
                            ]),
                            transform2=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std),
                            ]),
                            transform3=transforms.Compose1([
                              transforms.Scale1(384, 384),
                              transforms.ToTensor1(),
                              transforms.Normalize1(mean, std),
                            ]),
                            seed=seed,
                            pre=pre,
                            label_file=label_file)
  elif split == 'train':
    dataset = commonDataset(root=dataroot,
                            transform=transforms.Compose([
                              transforms.Scale(originalSize_h, originalSize_w),
                              transforms.RandomCrop(imageSize_h, imageSize_w),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std),
                            ]),
                            seed=seed,
                            pre=pre,
                            label_file=label_file)
  elif split == 'LRN_train_guide2':
    dataset = commonDataset(root=dataroot,
                            transform1=transforms.Scale(originalSize_h, originalSize_w),
                            transform2=transforms.RandomCrop_index(imageSize_h, imageSize_w),
                            transform3=transforms.RandomHorizontalFlip_index(),
                            transform4=transforms.GuideCrop(384, 384),
                            transform5=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std),
                            ]),
                            transform6=transforms.Compose1([
                              transforms.Scale1(384, 384),
                              transforms.ToTensor1(),
                              transforms.Normalize1(mean, std),
                            ]),
                            seed=seed,
                            pre=pre,
                            list_file=list_file)
  elif split == 'LRN_train_guide':
    dataset = commonDataset(root=dataroot,
                            transform1=transforms.Scale(originalSize_h, originalSize_w),
                            transform2=transforms.RandomCrop_index(imageSize_h, imageSize_w),
                            transform3=transforms.RandomHorizontalFlip_index(),
                            transform4=transforms.GuideCrop(384, 384),
                            transform5=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std),
                            ]),
                            transform6=transforms.Compose1([
                              transforms.Scale1(384, 384),
                              transforms.ToTensor1(),
                              transforms.Normalize1(mean, std),
                            ]),
                            seed=seed,
                            pre=pre,
                            label_file=label_file)

  dataloader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=batchSize, 
                                           shuffle=shuffle, 
                                           num_workers=int(workers))
  return dataloader


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
      self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


import numpy as np
class ImagePool:
  def __init__(self, pool_size=50):
    self.pool_size = pool_size
    if pool_size > 0:
      self.num_imgs = 0
      self.images = []

  def query(self, image):
    if self.pool_size == 0:
      return image
    if self.num_imgs < self.pool_size:
      self.images.append(image.clone())
      self.num_imgs += 1
      return image
    else:
      if np.random.uniform(0,1) > 0.5:
        random_id = np.random.randint(self.pool_size, size=1)[0]
        tmp = self.images[random_id].clone()
        self.images[random_id] = image.clone()
        return tmp
      else:
        return image


def adjust_learning_rate(optimizer, init_lr, epoch, factor, every):
  #import pdb; pdb.set_trace()
  lrd = init_lr / every
  old_lr = optimizer.param_groups[0]['lr']
   # linearly decaying lr
  lr = old_lr - lrd
  if lr < 0: lr = 0
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
