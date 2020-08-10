'''
Created on Sep 3, 2017

@author: Michal.Busta at gmail.com
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import LeakyReLU, Conv2d, Dropout2d, LogSoftmax, InstanceNorm2d
device = 'cuda'
import math


class _ConvBlock(nn.Sequential):
  def __init__(self, input_channel, growth_rate, dropout_rate=0.2):
    super(_ConvBlock, self).__init__()

    self.add_module('norm1_1', nn.BatchNorm2d(input_channel)),
    self.add_module('relu2_1', nn.ReLU(inplace=True)),
    self.add_module('conv2_1', nn.Conv2d(input_channel, growth_rate,
                                         kernel_size=3, stride=1, padding=1, bias=False))

    self.dropout_rate = dropout_rate

  def forward(self, x):
    new_features = super(_ConvBlock, self).forward(x)
    if self.dropout_rate > 0:
      new_features = F.dropout(new_features, p=self.dropout_rate, training=self.training)

    return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
  def __init__(self, nb_layers, nb_filter, growth_rate, dropout_rate=0.2, weight_decay=1e-4):
    super(_DenseBlock, self).__init__()

    for i in range(nb_layers):
      layer = _ConvBlock(nb_filter + i * growth_rate, growth_rate, dropout_rate)
      self.add_module('conv_block%d' % (i + 1), layer)


class _TransitionBlock(nn.Sequential):
  def __init__(self, nb_in_filter, nb_out_filter, dropout_rate=None):
    super(_TransitionBlock, self).__init__()

    self.add_module('norm', nn.BatchNorm2d(nb_in_filter))
    self.add_module('relu', nn.ReLU(inplace=True))
    self.add_module('conv', nn.Conv2d(nb_in_filter, nb_out_filter, kernel_size=1, stride=1, bias=False))

    self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class Densenet(nn.Module):
  def __init__(self, dropout_rate):
    super(Densenet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False)

    self.dense_block1 = _DenseBlock(nb_layers=4, nb_filter=64, growth_rate=16, dropout_rate=dropout_rate)
    self.trans_block1 = _TransitionBlock(nb_in_filter=64 + 16 * 4, nb_out_filter=128)

    self.dense_block2 = _DenseBlock(nb_layers=6, nb_filter=128, growth_rate=16, dropout_rate=dropout_rate)
    self.trans_block2 = _TransitionBlock(nb_in_filter=128 + 16 * 6, nb_out_filter=224)

    self.dense_block3 = _DenseBlock(nb_layers=4, nb_filter=224, growth_rate=16, dropout_rate=dropout_rate)

    self.batch_norm4 = nn.BatchNorm2d(288)

  def forward(self, input):
    out = self.conv1(input[:, :, :, :] - 0.5)

    out = self.dense_block1(out)
    out = self.trans_block1(out)

    out = self.dense_block2(out)
    out = self.trans_block2(out)

    out = self.dense_block3(out)

    src = F.relu(self.batch_norm4(out), inplace=True)
    return src


class BidirectionalLSTM(nn.Module):
  def __init__(self, num_features, hidden_size, output_size):
    super(BidirectionalLSTM, self).__init__()

    self.rnn = nn.LSTM(num_features, hidden_size, bidirectional=True)
    self.output = nn.Linear(hidden_size * 2, output_size)

  def forward(self, input_features):
    recurrent, _ = self.rnn(input_features)
    T, b, h = recurrent.size()
    t_rec = recurrent.view(T * b, h)

    output = self.output(t_rec)  # [T * b, nOut]
    output = output.view(T, b, -1)

    return output


class Encoder(nn.Module):
  def __init__(self, dropout_rate=0.2, net='densenet'):
    super(Encoder, self).__init__()
    if net == 'densenet':
      self.model = Densenet(dropout_rate=dropout_rate)
    # elif net == 'efficientnet':
      # self.model = EfficientNet.from_name('efficientnet-b0')

  def forward(self, input):
    out = self.model(input)
    return out


class Decoder(nn.Module):
  def __init__(self, input_dim, hidden_dim=512, num_class=200, relation_aware=False, net='densenet'):
    super(Decoder, self).__init__()
    if net == 'densenet':
      input_dim = 288 * 6
    elif net == 'efficientnet':
      input_dim = 320 * 2
    self.rnn1 = BidirectionalLSTM(input_dim, hidden_dim, hidden_dim)
    self.rnn2 = BidirectionalLSTM(hidden_dim, hidden_dim, num_class)

  def forward(self, X):
    X = X.contiguous().view(X.size(0), X.size(1) * X.size(2), X.size(3))
    X = X.permute(2, 0, 1).contiguous()
    X = self.rnn1(X)
    output = self.rnn2(X)
    return output


class Model(nn.Module):
  def __init__(self, num_classes, fixed_height=48, net='densenet'):
    super(Model, self).__init__()
    self.encoder = Encoder(net=net)
    self.decoder = Decoder(input_dim=int(fixed_height * 288 / 8), num_class=num_classes, net=net)
    self.crnn = nn.Sequential(
      self.encoder,
      self.decoder
    )
    self.log_softmax = nn.LogSoftmax(dim=2)

  def forward(self, input):
    output = self.crnn(input)
    output = self.log_softmax(output)
    output.permute(1, 2, 0)

    return output


class CReLU(nn.Module):
  def __init__(self):
    super(CReLU, self).__init__()
  def forward(self, x):
    return torch.cat((F.leaky_relu(x, 0.01, inplace=True), F.leaky_relu(-x, 0.01, inplace=True)), 1)

class CReLU_IN(nn.Module):
  def __init__(self, channels):
    super(CReLU_IN, self).__init__()
    self.bn = nn.InstanceNorm2d(channels * 2, eps=1e-05, momentum=0.1, affine=True)
  def forward(self, x):
    cat = torch.cat((x, -x), 1)
    x = self.bn(cat)
    return F.leaky_relu(x, 0.01, inplace=True)



def conv_bn(inp, oup, stride):
    return nn.Sequential(
      nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
      nn.BatchNorm2d(oup),
      nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride, dilation=1):
  return nn.Sequential(
    nn.Conv2d(inp, inp, 3, stride, 1 + (dilation > 0) * (dilation -1), dilation=dilation, groups=inp, bias=False),
    nn.BatchNorm2d(inp),
    nn.LeakyReLU(inplace=True, negative_slope=0.01),

    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
    nn.BatchNorm2d(oup),
    nn.LeakyReLU(inplace=True, negative_slope=0.01),
  )
  
def conv_dw_plain(inp, oup, stride, dilation=1):
  return nn.Sequential(
    nn.Conv2d(inp, inp, 3, stride, 1 + (dilation > 0) * (dilation -1), dilation=dilation, groups=inp, bias=False),
    nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
  )
  
def conv_dw_res(inp, oup, stride):
  return nn.Sequential(
    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
    nn.BatchNorm2d(inp),
    nn.LeakyReLU(inplace=True, negative_slope=0.01),

    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
    nn.BatchNorm2d(oup),
  )

def conv_dw_in(inp, oup, stride, dilation=1):
  return nn.Sequential(
    nn.Conv2d(inp, inp, 3, stride, 1 + (dilation > 0) * (dilation -1), dilation=dilation, groups=inp, bias=False),
    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
    InstanceNorm2d(oup, eps=1e-05, momentum=0.1),
    nn.LeakyReLU(inplace=True, negative_slope=0.01),
  )

def conv_dw_res_in(inp, oup, stride):
  return nn.Sequential(
    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
    nn.InstanceNorm2d(inp, eps=1e-05, momentum=0.1, affine=True),
    nn.LeakyReLU(inplace=True, negative_slope=0.01),

    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
    nn.InstanceNorm2d(oup, eps=1e-05, momentum=0.1, affine=True)
  )
    
def dice_loss(inp, target):
    
  smooth = 1.
  iflat = inp.view(-1)
  tflat = target.view(-1)
  intersection = (iflat * tflat).sum()

  return - ((2. * intersection + smooth) /
            (iflat.sum() + tflat.sum() + smooth))

class BasicBlockSep(nn.Module):
  expansion = 1
  def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
    super(BasicBlockSep, self).__init__()
    self.conv_sep1 = conv_dw(inplanes, planes, stride, dilation=dilation)
    self.conv2 = conv_dw_res(planes, planes, 1)
    self.downsample = downsample
    self.stride = stride
    self.relu = LeakyReLU(negative_slope=0.01, inplace=True)

  def forward(self, x):
    residual = x

    out = self.conv_sep1(x)

    out = self.conv2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out  
      
class BasicBlockIn(nn.Module):
  expansion = 1
  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlockIn, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = InstanceNorm2d(planes, eps=1e-05, momentum=0.1, affine=True)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = InstanceNorm2d(planes, eps=1e-05, momentum=0.1, affine=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out
  
class BasicBlockSepIn(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
    super(BasicBlockSepIn, self).__init__()
    
    self.conv_sep1 = conv_dw_in(inplanes, planes, stride, dilation=dilation)
    self.conv2 = conv_dw_res_in(planes, planes, 1)
    self.downsample = downsample
    self.stride = stride
    self.relu = LeakyReLU(negative_slope=0.01, inplace=True)

  def forward(self, x):
    residual = x

    out = self.conv_sep1(x)

    out = self.conv2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out  
  
def iou_loss(roi_gt, byte_mask, roi_pred):
# def iou_loss(roi_gt, byte_mask, roi_pred, box_loss_value):
  d1_gt = roi_gt[:, :, :, 0][byte_mask]
  d2_gt = roi_gt[:, :, :, 1][byte_mask] 
  d3_gt = roi_gt[:, :, :, 2][byte_mask]
  d4_gt = roi_gt[:, :, :, 3][byte_mask] 
  
  mask3 = torch.gt(d3_gt, 0)   
  mask4 = torch.gt(d4_gt, 0)   
  d3_gt = d3_gt[mask3]
  d4_gt = d4_gt[mask4] 
  
  
  d1_pred = roi_pred[:, 0, :, :][byte_mask]
  d2_pred = roi_pred[:, 1, :, :][byte_mask]
  d3_pred = roi_pred[:, 2, :, :][byte_mask]
  d3_pred = d3_pred[mask3]
  d4_pred = roi_pred[:, 3, :, :][byte_mask]
  d4_pred = d4_pred[mask4]

  #? LEFT
  #? d1 + d2 = height ____ d3 = width (left)
  area_gt_l = (d1_gt[mask3] + d2_gt[mask3]) * (d3_gt)
  area_pred_l = (d1_pred[mask3] + d2_pred[mask3]) * (d3_pred)
#union - phép hợp
  w_union_l = torch.min(d3_gt, d3_pred)
  h_union_l = torch.min(d1_gt[mask3], d1_pred[mask3]) + torch.min(d2_gt[mask3], d2_pred[mask3])

  #intersect - phép giao
  area_intersect_l = w_union_l * h_union_l
  area_union_l = area_gt_l + area_pred_l - area_intersect_l
  AABB_l = - torch.log((area_intersect_l + 1.0)/(area_union_l + 1.0))
  
  if AABB_l.dim() > 0:
  #? bug
    # box_loss_value += torch.mean(AABB_l)
    box_loss_value = torch.mean(AABB_l)
  #? d1 + d2 = width ____ d4 = height (right)
  area_gt_r = (d1_gt[mask4] + d2_gt[mask4]) * (d4_gt)
  area_pred_r = (d1_pred[mask4] + d2_pred[mask4]) * (d4_pred)

  w_union_r = torch.min(d4_gt, d4_pred)
  h_union_r = torch.min(d1_gt[mask4], d1_pred[mask4]) + torch.min(d2_gt[mask4], d2_pred[mask4])

  area_intersect_r = w_union_r * h_union_r
  area_union_r = area_gt_r + area_pred_r - area_intersect_r
  AABB_r = - torch.log((area_intersect_r + 1.0)/(area_union_r + 1.0))
  if AABB_r.dim() > 0:
    box_loss_value += torch.mean(AABB_r)
  return box_loss_value
  

class ModelResNetSep_crnn(nn.Module):

  def recompute(self):
    self.layer0[0].recompute_weights()
    self.layer0[2].recompute_weights()
    self.layer0_1[0].recompute_weights()
    self.layer0_1[2].recompute_weights()

  def __init__(self, attention = False, multi_scale = True,num_classes = 400 ,fixed_height=48, net = 'densenet',):
    super(ModelResNetSep_crnn, self).__init__()

    self.inplanes = 64

    self.layer0 = nn.Sequential(
      Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
      CReLU_IN(16),
      Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
      CReLU_IN(32)
    )

    self.layer0_1 = nn.Sequential(
      Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
      #nn.InstanceNorm2d(64, affine=True),
      nn.ReLU(),
      Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
      #nn.InstanceNorm2d(64, affine=True),
      nn.ReLU(inplace=True)
    )
##############3
    self.encoder = Encoder(net=net)
    self.decoder = Decoder(input_dim=int(fixed_height * 288 / 8), num_class=num_classes, net=net)
    self.crnn = nn.Sequential(
      self.encoder,
      self.decoder
    )
    self.log_softmax = nn.LogSoftmax(dim=2)
##########3


    # self.conv5 = Conv2d(64, 128, (3,3), padding=(1, 1), bias=False)
    # self.conv6 = Conv2d(128, 128, (3,3), padding=1, bias=False)
    # self.conv7 = Conv2d(128,256, 3, padding=1, bias=False)
    # self.conv8 = Conv2d(256, 256, (3,3), padding=1, bias=False)
    # self.conv9 = Conv2d(256, 256, (3,3), padding=(1, 1), bias=False)
    # self.conv10_s = Conv2d(256, 256, (2, 3), padding=(0, 1), bias=False)
    # self.conv11_fix = Conv2d(256, 400, (1, 1), padding=(0,0))
    #
    # self.batch5 = InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
    # self.batch6 = InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
    # self.batch7 = InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    # self.batch8 = InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    # self.batch9 = InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    # self.batch10_s = InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    # self.max2 = nn.MaxPool2d((2, 1), stride=(2,1))
    # self.leaky = LeakyReLU(negative_slope=0.01, inplace=True)

    self.layer1 = self._make_layer(BasicBlockIn, 64, 3, stride=1)
    self.inplanes = 64
    self.layer2 = self._make_layer(BasicBlockIn, 128, 4, stride=2)
    self.layer3 = self._make_layer(BasicBlockSepIn, 256, 6, stride=2)
    self.layer4 = self._make_layer(BasicBlockSepIn, 512, 4, stride=2)

    self.feature4 = nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False)
    self.feature3 = nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False)
    self.feature2 = nn.Conv2d(128, 256, 1, stride=1, padding=0, bias=False)

    self.upconv2 = conv_dw_plain(256, 256, stride=1)
    self.upconv1 = conv_dw_plain(256, 256, stride=1)

    self.feature1 = nn.Conv2d(64, 256, 1, stride=1, padding=0, bias=False)

    self.act = Conv2d(256, 1, (1,1), padding=0, stride=1) #seg
    self.rbox = Conv2d(256, 4, (1,1), padding=0, stride=1) #roi

    self.angle = Conv2d(256, 2, (1,1), padding=0, stride=1) #angle
    self.drop1 = Dropout2d(p=0.2, inplace=False)

    self.angle_loss = nn.MSELoss(reduction='elementwise_mean')
    self.h_loss = nn.SmoothL1Loss(reduction='elementwise_mean')
    self.w_loss = nn.SmoothL1Loss(reduction='elementwise_mean')

    self.attention = attention

    if self.attention:
      self.conv_attenton = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=True)

    self.multi_scale = multi_scale

  def _make_layer(self, block, planes, blocks, stride=1):

    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(

        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward_ocr(self, x):

    output = self.crnn(x)
    output = self.log_softmax(output)
    # output = output.permute(1, 2, 0)

    return output

    # x = x.squeeze(2)
    #
    # x = x.permute(0,2,1)
    # y = x
    # x = x.contiguous().view(-1,x.data.shape[2])
    # x = LogSoftmax(len(x.size()) - 1)(x)
    # x = x.view_as(y)
    # x = x.permute(0,2,1)

    return x

  # def forward_features(self, x):
  #
  #   x = self.layer0(x)
  #   focr = self.layer0_1(x)
  #   return focr

  def forward(self, x):

    x = self.layer0(x)
    x = self.layer0_1(x)

    x = self.drop1(x)
    su3 = self.layer1(x)
    features1 = self.feature1(su3)
    su2 = self.layer2(su3)
    features2 = self.feature2(su2)
    su1 = self.layer3(su2)
    features3 = self.feature3(su1)
    x = self.layer4(su1)

    x = self.drop1(x)

    features4 = self.feature4(x)
    if self.attention:
      att = self.conv_attenton(features4)
      att = torch.sigmoid(att)
      att = att.expand_as(features4)
      att_up = F.interpolate(att, size=(features3.size(2), features3.size(3)), mode='bilinear', align_corners=True)

    x = F.interpolate(features4, size=(features3.size(2), features3.size(3)), mode='bilinear', align_corners=True)

    if self.attention:
      x = x + features3 * att_up
      att = self.conv_attenton(x)
      att = torch.sigmoid(att)
      att_up = F.interpolate(att, size=(features2.size(2), features2.size(3)), mode='bilinear', align_corners=True)
    else:
      x = x + features3

    x = F.interpolate(x, size=(features2.size(2), features2.size(3)), mode='bilinear', align_corners=True)
    x = self.upconv1(x)
    if self.attention:
      features2 = x + features2 * att_up
      att = self.conv_attenton(features2)
      att = torch.sigmoid(att)
      att_up = F.interpolate(att, size=(features1.size(2), features1.size(3)), mode='bilinear', align_corners=True)
    else:
      features2 = x + features2
    x = features2

    x = F.interpolate(x, size=(features1.size(2), features1.size(3)), mode='bilinear', align_corners=True)
    x = self.upconv2(x)

    if self.attention:
      x = x + features1 * att_up
    else:
      x += features1

    segm_pred2 = torch.sigmoid(self.act(features2))
    rbox2 = torch.sigmoid(self.rbox(features2)) * 128
    angle2 = torch.sigmoid(self.angle(features2)) * 2 - 1
    angle_den = torch.sqrt(angle2[:, 0, :, :] * angle2[:, 0, :, :] + angle2[:, 1, :, :] * angle2[:, 1, :, :]).unsqueeze(1)
    angle_den = angle_den.expand_as(angle2)
    angle2 = angle2 / angle_den

    x = self.drop1(x)

    segm_pred = torch.sigmoid(self.act(x))
    rbox = torch.sigmoid(self.rbox(x)) * 128
    angle = torch.sigmoid(self.angle(x)) * 2 - 1
    angle_den = torch.sqrt(angle[:, 0, :, :] * angle[:, 0, :, :] + angle[:, 1, :, :] * angle[:, 1, :, :]).unsqueeze(1)
    angle_den = angle_den.expand_as(angle)
    angle = angle / angle_den

    return [segm_pred, segm_pred2], [rbox, rbox2], [angle, angle2], x

  def loss(self, segm_preds, segm_gt, iou_mask, angle_preds, angle_gt, roi_pred, roi_gt):

    self.box_loss_value =  torch.tensor(0.0, requires_grad = True).to(device)
    self.angle_loss_value =  torch.tensor(0.0, requires_grad = True).to(device)
    # print(self.angle_loss_value.is_leaf)
    # print(self.box_loss_value.is_leaf)
#loss  score (segm)
    segm_pred = segm_preds[0].squeeze(1)
    segm_pred1 = segm_preds[1].squeeze(1)
    angle_pred = angle_preds[0]
    self.segm_loss_value = dice_loss(segm_pred * iou_mask , segm_gt * iou_mask )

    if self.multi_scale:
      iou_gts = F.interpolate(segm_gt.unsqueeze(1), size=(segm_pred1.size(1), segm_pred1.size(2)), mode='bilinear', align_corners=True).squeeze(1)
      iou_masks = F.interpolate(iou_mask.unsqueeze(1), size=(segm_pred1.size(1), segm_pred1.size(2)), mode='bilinear', align_corners=True).squeeze(1)
      self.segm_loss_value += dice_loss(segm_pred1 * iou_masks, iou_gts * iou_masks )
# byte mask là mặt nạ nơi tọa độ có score>0.5
    byte_mask = torch.gt(segm_gt, 0.5)
#? chỉ những chỗ nào có (score_gt/segm_gt) thì mới tính angle tại tọa độ điểm đó, tượng tự với cả loss iou
    if byte_mask.sum() > 0:

      gt_sin = torch.sin(angle_gt[byte_mask])
      gt_cos = torch.cos(angle_gt[byte_mask])

      sin_val = self.angle_loss(angle_pred[:, 0, :, :][byte_mask], gt_sin)
      cos_val = self.angle_loss(angle_pred[:, 1, :, :][byte_mask], gt_cos)
#? bug leaf variance
      #self.angle_loss_value += sin_val
      # angleore_loss = tổng sinloss v cosloss
      self.angle_loss_value =self.angle_loss_value + sin_val
      self.angle_loss_value += cos_val

      # iou_loss(roi_gt, byte_mask, roi_pred[0], self.box_loss_value)
      self.box_loss_value = iou_loss(roi_gt, byte_mask, roi_pred[0])

      if self.multi_scale:
        byte_mask = torch.gt(F.interpolate(segm_gt.unsqueeze(1), size=(segm_pred1.size(1), segm_pred1.size(2)), mode='bilinear', align_corners=True), 0.5).squeeze(1)
        if byte_mask.sum() > 0:

          angle_gts = F.interpolate(angle_gt.unsqueeze(1), size=(segm_pred1.size(1), segm_pred1.size(2)), mode='bilinear', align_corners=True).squeeze(1)
          gt_sin = torch.sin(angle_gts[byte_mask])
          gt_cos = torch.cos(angle_gts[byte_mask])
          sin_val = self.angle_loss(angle_preds[1][:, 0, :, :][byte_mask], gt_sin)

          self.angle_loss_value += sin_val
          # self.angle_loss_value += sin_val
          self.angle_loss_value += self.angle_loss(angle_preds[1][:, 1, :, :][byte_mask], gt_cos)

          roi_gt_s = F.interpolate(roi_gt.permute(0, 3, 1, 2), size=(segm_pred1.size(1), segm_pred1.size(2)), mode='bilinear', align_corners=True) / 2
          roi_gt_s = roi_gt_s.permute(0, 2, 3, 1)
          self.box_loss_value += iou_loss(roi_gt_s, byte_mask, roi_pred[1])

    return self.segm_loss_value +  self.angle_loss_value * 2 + 0.5 * self.box_loss_value


