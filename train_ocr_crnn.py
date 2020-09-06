'''
Created on Sep 29, 2017

@author: Michal.Busta at gmail.com
'''
import numpy as np
import torch.nn.functional as F
import os
import torch
import net_utils
import argparse
import time
import ocr_gen
import torch.nn as nn
from models_crnn import ModelResNetSep_crnn
from ocr_test_utils import print_seq_ext
from utils import E2Ecollate,E2Edataset,alignCollate,ocrDataset
from torchvision import transforms
from net_eval import strLabelConverter,eval_ocr_crnn
import matplotlib.pyplot as plt

device = 'cuda'
f = open('codec.txt', 'r')
codec = f.readlines()[0]
f.close()
print(len(codec))
base_lr = 0.001
lr_decay = 0.99
momentum = 0.9
weight_decay = 0.0005
batch_per_epoch = 1000
disp_interval = 200

     
def main(opts):

  train_loss = 0
  train_loss_lr = 0
  cnt = 1
  cntt = 0
  time_total = 0
  now = time.time()
  converter = strLabelConverter(codec)

  model_name = 'E2E-MLT'
  net = ModelResNetSep_crnn(attention=True, multi_scale=True, num_classes=400, fixed_height=opts.norm_height,
                             net='densenet', )
  ctc_loss = nn.CTCLoss()
  optimizer = torch.optim.Adam(net.parameters(), lr=base_lr, weight_decay=weight_decay)
  # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5 ,patience=5,verbose=True)
  scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00007, max_lr=0.0003, step_size_up=3000,
                                                cycle_momentum=False)
  step_start = 0
  if opts.cuda:
    net.to(device)
    ctc_loss.to(device)
  if os.path.exists(opts.model):
    print('loading model from %s' % args.model)
    step_start, learning_rate = net_utils.load_net(args.model, net, optimizer)
  else:
    learning_rate = base_lr

  for param_group in optimizer.param_groups:
    param_group['lr'] = base_lr
    learning_rate = param_group['lr']
    print(param_group['lr'])

  step_start = 0

  net.train()

  # data_generator = ocr_gen.get_batch(num_workers=opts.num_readers,
  #                                    batch_size=opts.batch_size,
  #                                    train_list=opts.train_list, in_train=True, norm_height=opts.norm_height, rgb = True)

  data_dataset = ocrDataset(root=opts.train_list, norm_height=opts.norm_height , in_train=True)
  data_generator1 = torch.utils.data.DataLoader(data_dataset, batch_size=opts.batch_size, shuffle=True,
                                             collate_fn=alignCollate())
  val_dataset = ocrDataset(root=opts.valid_list, norm_height=opts.norm_height , in_train=False)
  val_generator1 = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             collate_fn=alignCollate())




  for step in range(step_start, 300000):
     # images, labels, label_length = next(data_generator)
     # im_data = net_utils.np_to_variable(images, is_cuda=opts.cuda).permute(0, 3, 1, 2)

    try:
       images, label = next(dataloader_iterator)
    except:
       dataloader_iterator = iter(data_generator1)
       images, label = next(dataloader_iterator)
    labels, label_length = converter.encode(label)
    im_data = images.to(device)
    labels_pred = net.forward_ocr(im_data)

    # backward
    probs_sizes =  torch.IntTensor( [(labels_pred.size()[0])] * (labels_pred.size()[1]) )
    label_sizes = torch.IntTensor( torch.from_numpy(np.array(label_length)).int() )
    labels = torch.IntTensor( torch.from_numpy(np.array(labels)).int() )
    loss = ctc_loss(labels_pred, labels, probs_sizes, label_sizes) / im_data.size(0) # change 1.9.
    optimizer.zero_grad()
    loss.backward()

    clipping_value = 1.0
    torch.nn.utils.clip_grad_norm_(net.parameters(),clipping_value)
    if not (torch.isnan(loss) or torch.isinf(loss)):
      optimizer.step()
      scheduler.step()
      train_loss += loss.data.cpu().numpy() #net.bbox_loss.data.cpu().numpy()[0]
      # train_loss += loss.data.cpu().numpy()[0] #net.bbox_loss.data.cpu().numpy()[0]
      cnt += 1

    if opts.debug:
      dbg = labels_pred.permute(1, 2, 0).data.cpu().numpy()
      ctc_f = dbg.swapaxes(1, 2)
      labels = ctc_f.argmax(2)
      det_text, conf, dec_s,_ = print_seq_ext(labels[0, :], codec)

      print('{0} \t'.format(det_text))



    if step % disp_interval == 0:
      for param_group in optimizer.param_groups:
        learning_rate = param_group['lr']
      train_loss /= cnt
      train_loss_lr += train_loss
      cntt += 1
      time_now = time.time() - now
      time_total += time_now
      now = time.time()
      save_log = os.path.join(opts.save_path, 'loss_ocr.txt')
      # f = open('content/drive/My_Drive/DATA_OCR/backup/ca ca/loss.txt','a')
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


      for param_group in optimizer.param_groups:
        learning_rate = param_group['lr']
        # print(learning_rate)

      save_name = os.path.join(opts.save_path, 'OCR_{}_{}.h5'.format(model_name, step))
      state = {'step': step,
               'learning_rate': learning_rate,
               'state_dict': net.state_dict(),
               'optimizer': optimizer.state_dict()}
      torch.save(state, save_name)
      # scheduler.step(train_loss_lr / cntt)
      # evaluate
      CER, WER = eval_ocr_crnn(val_generator1, net)
      # scheduler.step(CER)
      f = open(save_log, 'a')
      f.write('time epoch [%d]: %.2f s, loss_total: %.3f, CER = %f, WER = %f' % (step / batch_per_epoch, time_total, train_loss_lr / cntt, CER, WER))
      f.close()
      print('time epoch [%d]: %.2f s, loss_total: %.3f, CER = %f, WER = %f \n' % (step / batch_per_epoch, time_total, train_loss_lr / cntt, CER, WER))
      print('save model: {}'.format(save_name))
      net.train()
      time_total = 0
      cntt = 0
      train_loss_lr = 0

if __name__ == '__main__': 
  parser = argparse.ArgumentParser()
  
  parser.add_argument('-train_list', default='/content/data_MLT_crop/gt_vi.txt')
  parser.add_argument('-valid_list', default='/content/data_MLT_crop/gt_vi_eval.txt')
  parser.add_argument('-save_path', default='/content/drive/My Drive/DATA_OCR/ocr_lstm')
  parser.add_argument('-model', default='E2E-MLT_69000.h5')
  parser.add_argument('-debug', type=int, default=0)
  parser.add_argument('-batch_size', type=int, default=8)
  parser.add_argument('-num_readers', type=int, default=2)
  parser.add_argument('-cuda', type=bool, default=True)
  parser.add_argument('-norm_height', type=int, default=64)
  
  args = parser.parse_args()  
  main(args)
  
