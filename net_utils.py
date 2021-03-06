'''
Created on Aug 31, 2017

@author: Michal.Busta at gmail.com
'''
import numpy as np
import torch
from torch.autograd import Variable

def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
  v = torch.from_numpy(x).type(dtype)
  if is_cuda:
      v = v.cuda()
  return v

def load_net(fname, net, optimizer=None):
#   sp = torch.load(fname,map_location=torch.device('cpu'))
  sp = torch.load(fname) 
  step = sp['step']
  try:
    learning_rate = sp['learning_rate']
  except:
    import traceback
    traceback.print_exc()
    learning_rate = 0.001
  opt_state = sp['optimizer']
  sp = sp['state_dict']
  for k, v in net.state_dict().items() :
    try:
      if (k in sp) and (sp[k].size() == v.size()):
        param = sp[k]
        v.copy_(param)
        # print(v.size())
      else:
        # print(k)
        # print(v.size())
        v.copy_(torch.randn(v.size()))
    except:
      import traceback
      traceback.print_exc()
  
  if optimizer is not None:  
    try:
      optimizer.load_state_dict(opt_state)
    except:
      import traceback
      traceback.print_exc()
  
  print(fname)
  return step, learning_rate 

def adjust_learning_rate(optimizer, lr):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  # lr = args.lr * (0.1 ** (epoch // 30))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  
  
