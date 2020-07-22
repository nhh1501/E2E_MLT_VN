'''
Created on Oct 25, 2018

@author: Michal.Busta at gmail.com
'''

import math
import numpy as np
import random
import torch
import torch.nn.functional as F
import cv2

device = 'cuda'

def print_seq_ext(wf, codec):
  prev = 0
  word = ''
  current_word = ''
  start_pos = 0
  end_pos = 0
  dec_splits = []
  splits = []
  hasLetter = False
  for cx in range(0, wf.shape[0]):
    c = wf[cx]
    if prev == c:
      if c > 2:
          end_pos = cx
      continue
    if c > 3 and c < (len(codec)+4):
      ordv = codec[c - 4]
      char = ordv
      if char == ' ' or char == '.' or char == ',' or char == ':':
        if hasLetter:
          if char != ' ':
            current_word += char
          splits.append(current_word)
          dec_splits.append(cx + 1)
          word += char
          current_word = ''
      else:
        hasLetter = True
        word += char
        current_word += char
      end_pos = cx
    elif c > 0:
      if hasLetter:
        dec_splits.append(cx + 1)
        word += ' '
        end_pos = cx
        splits.append(current_word)
        current_word = ''
      
    
    if len(word) == 0:
      start_pos = cx
    prev = c    
    
  dec_splits.append(end_pos + 1)
  conf2 = [start_pos, end_pos + 1]
  
  return word.strip(), np.array([conf2]), np.array([dec_splits]), splits

def ocr_image(net, codec, im_data, detection,target_h):
  
  boxo = detection
  boxr = boxo[0:8].reshape(-1, 2)
  
  center = (boxr[0, :] + boxr[1, :] + boxr[2, :] + boxr[3, :]) / 4
  
  dw = boxr[2, :] - boxr[1, :]
  dh =  boxr[1, :] - boxr[0, :]

  w = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1])
  h = math.sqrt(dh[0] * dh[0] + dh[1] * dh[1]) + random.randint(-2, 2)
  
  input_W = im_data.size(3)
  input_H = im_data.size(2)
    
  scale = target_h / max(1, h) 
  target_gw = int(w * scale) + target_h 
  target_gw = max(2, target_gw // 32) * 32      
    
  xc = center[0] 
  yc = center[1] 
  w2 = w 
  h2 = h 
  
  angle = math.atan2((boxr[2][1] - boxr[1][1]), boxr[2][0] - boxr[1][0])
  
  #show pooled image in image layer

  scalex = (w2 + h2) / input_W * 1.2
  scaley = h2 / input_H * 1.3

  th11 =  scalex * math.cos(angle)
  th12 = -math.sin(angle) * scaley
  th13 =  (2 * xc - input_W - 1) / (input_W - 1) #* torch.cos(angle_var) - (2 * yc - input_H - 1) / (input_H - 1) * torch.sin(angle_var)
  
  th21 = math.sin(angle) * scalex 
  th22 =  scaley * math.cos(angle)  
  th23 =  (2 * yc - input_H - 1) / (input_H - 1) #* torch.cos(angle_var) + (2 * xc - input_W - 1) / (input_W - 1) * torch.sin(angle_var)
            
  t = np.asarray([th11, th12, th13, th21, th22, th23], dtype=np.float)
  t = torch.from_numpy(t).type(torch.FloatTensor)
  t = t.to(device)
  theta = t.view(-1, 2, 3)
  
  grid = F.affine_grid(theta, torch.Size((1, 3, int(target_h), int(target_gw))))
  
  
  x = F.grid_sample(im_data, grid)

  mask_gray = cv2.normalize(src=x.data.numpy().squeeze(), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_8UC1)
  cv2.imshow('abc', mask_gray.transpose(1, 2, 0))
  cv2.waitKey(0)
  

  features = net.forward_features(x)
  labels_pred = net.forward_ocr(features)
  
  ctc_f = labels_pred.data.cpu().numpy()
  ctc_f = ctc_f.swapaxes(1, 2)

  labels = ctc_f.argmax(2)
  
  ind = np.unravel_index(labels, ctc_f.shape)
  conf = np.mean( np.exp(ctc_f[ind]) )
  
  det_text, conf2, dec_s, splits = print_seq_ext(labels[0, :], codec)  
  
  return det_text, conf2, dec_s

def ocr_batch(net, codec, im_data, gtso,lbso,target_h):
#e2e
  num_gt = len(gtso)
  rrois = []
  labels = []
  for kk in range(num_gt):
    gts = gtso[kk]
    lbs = lbso[kk]
    if len(gts) != 0:
      gt = np.asarray(gts)
      # boxr = gts[0:8].reshape(-1, 2)

      center = (gt[:, 0, :] + gt[:, 1, :] + gt[:, 2, :] + gt[:, 3, :]) / 4  # 求中心点
      dw = gt[:, 2, :] - gt[:, 1, :]
      dh = gt[:, 1, :] - gt[:, 0, :]
      poww = pow(dw, 2)
      powh = pow(dh, 2)

      w = np.sqrt(poww[:, 0] + poww[:, 1])
      h = np.sqrt(powh[:, 0] + powh[:, 1]) + random.randint(-2, 2)

      # dw2= gt[:, 0, :] - gt[:, 3, :]
      # dh2= gt[:, 3, :] - gt[:, 2, :]
      # poww2 = pow(dw2, 2)
      # powh2 = pow(dh2, 2)
      # w2= np.sqrt(poww2[:, 0]  + poww2[:, 1])
      # h2= np.sqrt(powh2[:, 0] + powh2[:, 1]) + random.randint(-2, 2)
      #
      # w = (w + w2) / 2
      # h = (h + h2) / 2

      angle  = (np.arctan2((gt[:, 2, 1] - gt[:, 1, 1]), gt[:, 2, 0] - gt[:, 1, 0]))
      # angle2 = (np.arctan2((gt[:, 3, 1] - gt[:, 0, 1]), gt[:, 3, 0] - gt[:, 0, 0]))
      # angle = angle + angle2
      for gt_id in range(0, len(gts)):

        gt_txt = lbs[gt_id]  # 文字判断
        if gt_txt.startswith('##'):
          continue

        gt = gts[gt_id]  # 标注信息判断
        if gt[:, 0].max() > im_data.size(3) or gt[:, 1].max() > im_data.size(2) or gt.min() < 0:
          continue

        rrois.append([kk, center[gt_id][0], center[gt_id][1], h[gt_id], w[gt_id], angle[gt_id]])  # 将标注的rroi写入
        labels.append(gt_txt)

    # show pooled image in image layer

  rois = torch.tensor(rrois).to(torch.float)

  id_img = rois[:,0]
  xc = rois[:, 1]
  yc = rois[:, 2]
  h= rois[:,3]
  w= rois[:,4]
  angle = rois[:,5]
  input_W = im_data.size(3)
  input_H = im_data.size(2)

  scale = target_h / torch.max(torch.ones_like(h), h) # h = rois[:,3]
  target_gw = (w * scale) + target_h
  target_gw = torch.max(2*torch.ones_like(target_gw), target_gw // 32) * 32
  w2 = w
  h2 = h
  scalex = (w2 + h2) / input_W * 1.2
  scaley = h2 / input_H * 1.3

  th11 = scalex * torch.cos(angle)
  th12 = -torch.sin(angle) * scaley
  th13 = (2 * xc - input_W - 1) / (input_W - 1)  # * torch.cos(angle_var) - (2 * yc - input_H - 1) / (input_H - 1) * torch.sin(angle_var)

  th21 = torch.sin(angle) * scalex
  th22 = scaley * torch.cos(angle)
  th23 = (2 * yc - input_H - 1) / (input_H - 1)  # * torch.cos(angle_var) + (2 * xc - input_W - 1) / (input_W - 1) * torch.sin(angle_var)

  # t = np.asarray([th11, th12, th13, th21, th22, th23], dtype=np.float)
  # t = torch.from_numpy(t).type(torch.FloatTensor)
  t =torch.stack((th11, th12, th13, th21, th22, th23),-1)
  t = t.to(device)
  theta = t.view(-1, 2, 3)
  det_texts = []
  # labels_gt = []
  for idx in range(len(labels)):

    grid = F.affine_grid(theta[idx].unsqueeze(0), torch.Size((1, 3, int(target_h), int(target_gw[idx]))))

    x = F.grid_sample(im_data[int(id_img[idx])].unsqueeze(0), grid)


    # mask_gray = cv2.normalize(src=x.data.numpy().squeeze(), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
    #                           dtype=cv2.CV_8UC1)
    # cv2.imshow('abc',mask_gray.transpose(1, 2, 0))
    # cv2.waitKey(0)

    features = net.forward_features(x)
    labels_pred = net.forward_ocr(features)

    ctc_f = labels_pred.data.cpu().numpy()
    ctc_f = ctc_f.swapaxes(1, 2)

    labelss = ctc_f.argmax(2)

    ind = np.unravel_index(labelss, ctc_f.shape)
    # conf = np.mean(np.exp(ctc_f[ind]))

    det_text, conf2, dec_s, splits = print_seq_ext(labelss[0, :], codec)
    # if conf < 0.01 and len(det_text) == 3:
    #   print('Too low conf short: {0} {1}'.format(det_text, conf))
    #   continue
    det_texts.append(det_text)
    # labels_gt.append()
  return (det_texts, labels)

def crnn_batch(net, codec, im_data, gtso,lbso,target_h):

  num_gt = len(gtso)
  rrois = []
  labels = []
  for kk in range(num_gt):
    gts = gtso[kk]
    lbs = lbso[kk]
    if len(gts) != 0:
      gt = np.asarray(gts)
      # boxr = gts[0:8].reshape(-1, 2)

      center = (gt[:, 0, :] + gt[:, 1, :] + gt[:, 2, :] + gt[:, 3, :]) / 4  # 求中心点
      dw = gt[:, 2, :] - gt[:, 1, :]
      dh = gt[:, 1, :] - gt[:, 0, :]

      poww = pow(dw, 2)
      powh = pow(dh, 2)

      w = np.sqrt(poww[:, 0] + poww[:, 1])
      h = np.sqrt(powh[:, 0] + powh[:, 1]) + random.randint(-2, 2)

      # dw2= gt[:, 0, :] - gt[:, 3, :]
      # dh2= gt[:, 3, :] - gt[:, 2, :]
      #
      # poww2 = pow(dw2, 2)
      # powh2 = pow(dh2, 2)
      #
      # w2= np.sqrt(poww2[:, 0]  + poww2[:, 1])
      # h2= np.sqrt(powh2[:, 0] + powh2[:, 1]) + random.randint(-2, 2)
      #
      # w = (w + w2) / 2
      # h = (h + h2) / 2

      angle  = (np.arctan2((gt[:, 2, 1] - gt[:, 1, 1]), gt[:, 2, 0] - gt[:, 1, 0]))
      # angle2 = (np.arctan2((gt[:, 3, 1] - gt[:, 0, 1]), gt[:, 3, 0] - gt[:, 0, 0]))
      # angle = angle + angle2
      for gt_id in range(0, len(gts)):

        gt_txt = lbs[gt_id]  # 文字判断
        if gt_txt.startswith('##'):
          continue

        gt = gts[gt_id]  # 标注信息判断
        if gt[:, 0].max() > im_data.size(3) or gt[:, 1].max() > im_data.size(2) or gt.min() < 0:
          continue

        rrois.append([kk, center[gt_id][0], center[gt_id][1], h[gt_id], w[gt_id], angle[gt_id]])  # 将标注的rroi写入
        labels.append(gt_txt)

    # show pooled image in image layer

  rois = torch.tensor(rrois).to(torch.float)

  id_img = rois[:,0]
  xc = rois[:, 1]
  yc = rois[:, 2]
  h= rois[:,3]
  w= rois[:,4]
  angle = rois[:,5]
  input_W = im_data.size(3)
  input_H = im_data.size(2)
  # target_h = 44

  scale = target_h / torch.max(torch.ones_like(h), h) # h = rois[:,3]
  target_gw = (w * scale) + target_h
  target_gw = torch.max(2*torch.ones_like(target_gw), target_gw // 32) * 32
  w2 = w
  h2 = h
  scalex = (w2 + h2) / input_W * 1.2
  scaley = h2 / input_H * 1.3

  th11 = scalex * torch.cos(angle)
  th12 = -torch.sin(angle) * scaley
  th13 = (2 * xc - input_W - 1) / (input_W - 1)  # * torch.cos(angle_var) - (2 * yc - input_H - 1) / (input_H - 1) * torch.sin(angle_var)

  th21 = torch.sin(angle) * scalex
  th22 = scaley * torch.cos(angle)
  th23 = (2 * yc - input_H - 1) / (input_H - 1)  # * torch.cos(angle_var) + (2 * xc - input_W - 1) / (input_W - 1) * torch.sin(angle_var)

  # t = np.asarray([th11, th12, th13, th21, th22, th23], dtype=np.float)
  # t = torch.from_numpy(t).type(torch.FloatTensor)
  t =torch.stack((th11, th12, th13, th21, th22, th23),-1)
  t = t.to(device)
  theta = t.view(-1, 2, 3)
  det_texts = []
  # labels_gt = []
  for idx in range(len(labels)):

    grid = F.affine_grid(theta[idx].unsqueeze(0), torch.Size((1, 3, int(target_h), int(target_gw[idx]))))

    x = F.grid_sample(im_data[int(id_img[idx])].unsqueeze(0), grid)


    # mask_gray = cv2.normalize(src=x.data.numpy().squeeze(), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
    #                           dtype=cv2.CV_8UC1)
    # cv2.imshow('abc',mask_gray.transpose(1, 2, 0))
    # cv2.waitKey(0)

    # features = net.forward_features(x)
    labels_pred = net.forward_ocr(x)
    labels_pred = labels_pred.permute(1, 2, 0)

    ctc_f = labels_pred.data.cpu().numpy()
    ctc_f = ctc_f.swapaxes(1, 2)

    labelss = ctc_f.argmax(2)

    # ind = np.unravel_index(labelss, ctc_f.shape)
    # conf = np.mean(np.exp(ctc_f[ind]))

    det_text, conf2, dec_s, splits = print_seq_ext(labelss[0, :], codec)
    # if conf < 0.01 and len(det_text) == 3:
    #   print('Too low conf short: {0} {1}'.format(det_text, conf))
    #   continue
    det_texts.append(det_text)
    # labels_gt.append()
  return (det_texts, labels)
