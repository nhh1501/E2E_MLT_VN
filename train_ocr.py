'''
Created on Sep 29, 2017

@author: Michal.Busta at gmail.com
'''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import os

f = open('codec.txt', 'r')
codec = f.readlines()[0]
f.close()
print(len(codec))

import torch
import net_utils
import argparse

import ocr_gen
import time

# from warpctc_pytorch import CTCLoss
import torch.nn as nn
from torch.autograd import Variable
from utils import E2Ecollate,E2Edataset,alignCollate,ocrDataset
from models import  ModelResNetSep_final
from ocr_test_utils import print_seq_ext
from net_eval import eval_ocr
import random

import cv2
  

base_lr = 0.0001
lr_decay = 0.99
momentum = 0.9
weight_decay = 0.0005
batch_per_epoch = 15
disp_interval = 10

     
def main(opts):
  
  model_name = 'E2E-MLT'
  net = ModelResNetSep_final(attention=True)
  acc = []
  ctc_loss = nn.CTCLoss()
  if opts.cuda:
    net.cuda()
    ctc_loss.cuda()
  optimizer = torch.optim.Adam(net.parameters(), lr=base_lr, weight_decay=weight_decay)
  step_start = 0  
  if os.path.exists(opts.model):
    print('loading model from %s' % args.model)
    step_start, learning_rate = net_utils.load_net(args.model, net, optimizer)
  else:
    learning_rate = base_lr
  
  step_start = 0  

  net.train()
  
  #acc_test = test(net, codec, opts, list_file=opts.valid_list, norm_height=opts.norm_height)
  #acc.append([0, acc_test])
    
  # ctc_loss = CTCLoss()
  ctc_loss = nn.CTCLoss()

  data_generator = ocr_gen.get_batch(num_workers=opts.num_readers,
          batch_size=opts.batch_size, 
          train_list=opts.train_list, in_train=True, norm_height=opts.norm_height, rgb = True)
  
  val_dataset = ocrDataset(root=opts.valid_list, norm_height=opts.norm_height , in_train=False,is_crnn=False)
  val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False,
                                                collate_fn=alignCollate())


  # val_generator1 = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False,
  #                                              collate_fn=alignCollate())

  cnt = 1
  cntt = 0
  train_loss_lr = 0
  time_total = 0
  train_loss = 0
  now = time.time()

  for step in range(step_start, 300000):
    # batch
    images, labels, label_length = next(data_generator)
    im_data = net_utils.np_to_variable(images, is_cuda=opts.cuda).permute(0, 3, 1, 2)
    features = net.forward_features(im_data)
    labels_pred = net.forward_ocr(features)
    
    # backward
    '''
    acts: Tensor of (seqLength x batch x outputDim) containing output from network
        labels: 1 dimensional Tensor containing all the targets of the batch in one sequence
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        act_lens: Tensor of (batch) containing label length of each example
    '''
    
    probs_sizes =  torch.IntTensor( [(labels_pred.permute(2,0,1).size()[0])] * (labels_pred.permute(2,0,1).size()[1]) )
    label_sizes = torch.IntTensor( torch.from_numpy(np.array(label_length)).int() )
    labels = torch.IntTensor( torch.from_numpy(np.array(labels)).int() )
    loss = ctc_loss(labels_pred.permute(2,0,1), labels, probs_sizes, label_sizes) / im_data.size(0) # change 1.9.
    optimizer.zero_grad()
    loss.backward()

    clipping_value = 0.5
    torch.nn.utils.clip_grad_norm_(net.parameters(),clipping_value)
    if not (torch.isnan(loss) or torch.isinf(loss)):
      optimizer.step()
    # if not np.isinf(loss.data.cpu().numpy()):
      train_loss += loss.data.cpu().numpy() #net.bbox_loss.data.cpu().numpy()[0]
      # train_loss += loss.data.cpu().numpy()[0] #net.bbox_loss.data.cpu().numpy()[0]
      cnt += 1
    
    if opts.debug:
      dbg = labels_pred.data.cpu().numpy()
      ctc_f = dbg.swapaxes(1, 2)
      labels = ctc_f.argmax(2)
      det_text, conf, dec_s,_ = print_seq_ext(labels[0, :], codec)
      
      print('{0} \t'.format(det_text))
    
    
    
    if step % disp_interval == 0:
        
      train_loss /= cnt
      train_loss_lr += train_loss
      cntt += 1
      time_now = time.time() - now
      time_total += time_now
      now = time.time()
      save_log = os.path.join(opts.save_path, 'loss_ocr.txt')
      f = open(save_log, 'a')
      f.write(
        'epoch %d[%d], loss_ctc: %.3f,time: %.2f s, lr: %.5f, cnt: %d\n' % (
          step / batch_per_epoch, step, train_loss, time_now,learning_rate, cnt))
      f.close()

      print('epoch %d[%d], loss_ctc: %.3f,time: %.2f s, lr: %.5f, cnt: %d\n' % (
          step / batch_per_epoch, step, train_loss, time_now,learning_rate, cnt))

      train_loss = 0
      cnt = 1

    if step > step_start and (step % batch_per_epoch == 0):
      CER, WER = eval_ocr(val_generator, net)
      net.train()
      for param_group in optimizer.param_groups:
        learning_rate = param_group['lr']
        print(learning_rate)

      save_name = os.path.join(opts.save_path, '{}_{}.h5'.format(model_name, step))
      state = {'step': step,
               'learning_rate': learning_rate,
              'state_dict': net.state_dict(),
              'optimizer': optimizer.state_dict()}
      torch.save(state, save_name)
      print('save model: {}'.format(save_name))
      print('time epoch [%d]: %.2f s, loss_total: %.3f, CER = %f, WER = %f' % (
      step / batch_per_epoch, time_total, train_loss_lr / cntt, CER, WER))
      time_total = 0
      cntt = 0
      train_loss_lr = 0

      #acc_test, ted = test(net, codec, opts,  list_file=opts.valid_list, norm_height=opts.norm_height)
      #acc.append([0, acc_test, ted])
      # np.savez('train_acc_{0}'.format(model_name), acc=acc)

if __name__ == '__main__': 
  parser = argparse.ArgumentParser()
  
  parser.add_argument('-train_list', default='sample_train_data/MLT_CROPS/gt.txt')
  parser.add_argument('-valid_list', default='sample_train_data/MLT_CROPS/gt.txt')
  parser.add_argument('-save_path', default='backup2')
  parser.add_argument('-model', default='e2e-mlt.h5')
  parser.add_argument('-debug', type=int, default=1)
  parser.add_argument('-batch_size', type=int, default=1)
  parser.add_argument('-num_readers', type=int, default=1)
  parser.add_argument('-cuda', type=bool, default=True)
  parser.add_argument('-norm_height', type=int, default=44)
  
  args = parser.parse_args()  
  main(args)
  
