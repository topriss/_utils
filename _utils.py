import os
import sys
import time
import math
import random
from subprocess import Popen, PIPE
import json

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
  '''Compute running average.'''
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

def rupdate(d, u):
  for k, v in u.items():
    if isinstance(v, dict):
      d[k] = rupdate(d.get(k, {}), v)
    else:
      d[k] = v
  return d

def check_cfg_allset(cfg, k_prefix=''):
  flag = True
  for k, v in cfg.items():
    if isinstance(v, dict):
      flag &= check_cfg_allset(v, k)
    elif v is None:
      print(f'\'{k_prefix}.{k}\' is not set')
      flag = False
  return flag

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
  'jpg',
  'jpeg',
  'png',
  'ppm',
  'bmp',
  'pgm',
  'tif',
  'tiff',
  'webp',
)
def collect_images(rootdir, recursive=False):
  assert not recursive, 'recursive not longer supported'
  all_img = filter( lambda s:s.split('.')[-1].lower() in IMG_EXTENSIONS, os.listdir(rootdir) )
  return [os.path.realpath(os.path.join(rootdir, img)) for img in all_img]

def multipage(filename, dpi=100):
  print('preparing pdf ...')
  pp = PdfPages(filename)
  for fig in [plt.figure(n) for n in plt.get_fignums()]: fig.savefig(pp, dpi=dpi, format='pdf')
  pp.close()
  plt.close('all')

def read_img_np(img_fpath):
  img = Image.open(img_fpath).convert('RGB')
  return np.array(img, dtype=np.float32) / 255.0

def read_img_pt(img_fpath):
  img = Image.open(img_fpath).convert('RGB')
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

def torch_re(G=24-3):
  _r = torch.zeros([1024//4, 1024, 1024, 21], dtype=torch.float32, device='cuda')
  del _r

def fwd_anomaly_hook(self, m_in, m_out, out_idx_prefix=''):
  '''
  usage example:

  torch.set_anomaly_enabled(True)
  for n, m in netNloss.named_modules(prefix='netNloss'):
    m._debug_name = n
    m.register_forward_hook(fwd_anomaly_hook)
  '''
  if isinstance(m_out, torch.Tensor):
    # print(f'module {self._debug_name}, output {out_idx_prefix}, mean={m_out.mean().item():.2e}, var={m_out.var().item():.2e}')
    for ano_type in ['nan', 'inf']:
      if eval(f'm_out.is{ano_type}().any()'):
        print(m_out)
        print(m_out.shape)
        raise RuntimeError(f'module {self._debug_name}, output {out_idx_prefix}, found {ano_type}')
  elif isinstance(m_out, tuple):
    for i, _o in enumerate(m_out):
      fwd_anomaly_hook(self, m_in, _o, out_idx_prefix=out_idx_prefix+'.'+str(i))
  elif isinstance(m_out, dict):
    for k, v in m_out.items():
      fwd_anomaly_hook(self, m_in, v, out_idx_prefix=out_idx_prefix+'.'+k)
  else:
    print(f'in {self._debug_name}, m_out {out_idx_prefix}, unsupported m_out type {type(m_out)}')