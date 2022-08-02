import os
import sys
import time
import math
import random
from subprocess import Popen, PIPE

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class GO(object):
  def __init__(self, name=''):
    self.name = name
  def __enter__(self):
    pass
  def __exit__(self, type, value, traceback):
    pass

class AverageMeter(object):
  """Compute running average."""
  def __init__(self):
    self.val = 0
    self.sum = 0
    self.cnt = 0
    self.avg = 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def mkdir(work_dir):
  if os.path.isdir(work_dir):
    flag_overwrite = ''
    while flag_overwrite not in ['y', 'n']:
      flag_overwrite = input(f'overwriting existing dir {work_dir}, please confirm y/n: ')
    if flag_overwrite == 'y':
      os.system(f'rm -r {work_dir}')
    else:
      print('not overwriting, exiting')
      exit(0)
  else:
    print(f'creating dir {work_dir}')
  os.system(f'mkdir -p {work_dir}')

class Timer(object):
  def __init__(self, name=None):
    self.name = name
  def __enter__(self):
    self.tstart = time.time()
  def __exit__(self, type, value, traceback):
    print(f'{f"[{self.name}]" if self.name else ""} Elapsed: {time.time() - self.tstart:.3f}')

IMG_EXTENSIONS = (
  "jpg",
  "jpeg",
  "png",
  "ppm",
  "bmp",
  "pgm",
  "tif",
  "tiff",
  "webp",
)
def collect_images(rootdir, recursive=False):
  all_s = Popen(f"find {rootdir} -maxdepth {1000 if recursive else 1} -type f | sort",  shell=True, stdout=PIPE, stderr=PIPE).communicate()[0].decode("utf-8").strip().split('\n')
  return list(filter( lambda s:s.split('.')[-1].lower() in IMG_EXTENSIONS, all_s ))

def multipage(filename, dpi=100):
  print('preparing pdf ...')
  pp = PdfPages(filename)
  for fig in [plt.figure(n) for n in plt.get_fignums()]: fig.savefig(pp, dpi=dpi, format='pdf')
  pp.close()
  plt.close('all')

def read_img_pt(img_fpath):
  img = Image.open(img_fpath).convert("RGB")
  return transforms.ToTensor()(img)

class align_pad(object):
  def __init__(self, p=64):
    self.p = p
  def __call__(self, img):
    return self.pad(img)
  def __repr__(self):
    return f'alignpad({self.p})'
  def pad(self, img):
    h, w = img.shape[-2:]
    h_new = (h + self.p - 1) // self.p * self.p
    w_new = (w + self.p - 1) // self.p * self.p
    self.p_l = (w_new - w) // 2
    self.p_t = (h_new - h) // 2
    self.p_r = w_new - w - self.p_l
    self.p_b = h_new - h - self.p_t
    return F.pad(
      img,
      (self.p_l, self.p_r, self.p_t, self.p_b),
      mode='constant', value=0 )
  def unpad(self, img):
    return F.pad(
      img,
      (-self.p_l, -self.p_r, -self.p_t, -self.p_b) )
      
def torch_init(seed=11037):
  # torch.backends.cudnn.benchmark = True      # A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest. 
  # torch.backends.cudnn.deterministic = True  # A bool that, if True, causes cuDNN to only use deterministic convolution algorithms.

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)