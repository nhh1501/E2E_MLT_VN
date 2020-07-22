#!/usr/bin/python
# encoding: utf-8

import torch
import albumentations as A
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import collections
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import math
import random
import numpy as np
import cv2
import os
import PIL
from data_gen import generate_rbox, generate_rbox2
from data_gen import load_gt_annoataion
from data_gen import get_images
import torchvision.transforms as transforms
from data_gen import draw_box_points

# from rroi_align.modules.rroi_align import _RRoiAlign

# with open('./data/alphabet.txt', 'r') as f:
#     alphabet = f.readlines()[0]


# class strLabelConverter(object):
#     """Convert between str and label.
#
#     NOTE:
#         Insert `blank` to the alphabet for CTC.
#
#     Args:
#         alphabet (str): set of the possible characters.
#         ignore_case (bool, default=True): whether or not to ignore all of the case.
#     """
#
#     def __init__(self, alphabet, ignore_case=False):
#         self._ignore_case = ignore_case
#         if self._ignore_case:
#             alphabet = alphabet.lower()
#         self.alphabet = alphabet + '-'  # for `-1` index
#
#         self.dict = {}
#         for i, char in enumerate(alphabet):
#             # NOTE: 0 is reserved for 'blank' required by wrap_ctc
#             self.dict[char] = i + 1
#
#     def encode(self, text):
#         """Support batch or single str.
#
#         Args:
#             text (str or list of str): texts to convert.
#
#         Returns:
#             torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
#             torch.IntTensor [n]: length of each text.
#         """
#         if isinstance(text, str):
#             text = [
#                 self.dict[char.lower() if self._ignore_case else char]
#                 for char in text
#             ]
#             length = [len(text)]
#         elif isinstance(text, collections.Iterable):
#             length = [len(s) for s in text]
#             text = ''.join(text)
#             text, _ = self.encode(text)
#         return (torch.IntTensor(text), torch.IntTensor(length))
#
#     def decode(self, t, length, raw=False):
#         """Decode encoded texts back into strs.
#
#         Args:
#             torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
#             torch.IntTensor [n]: length of each text.
#
#         Raises:
#             AssertionError: when the texts and its length does not match.
#
#         Returns:
#             text (str or list of str): texts to convert.
#         """
#         if length.numel() == 1:
#             length = length[0]
#             assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
#             if raw:
#                 return ''.join([self.alphabet[i - 1] for i in t])
#             else:
#                 char_list = []
#                 for i in range(length):
#                     if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
#                         char_list.append(self.alphabet[t[i] - 1])
#                 return ''.join(char_list)
#         else:
#             # batch mode
#             assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
#             texts = []
#             index = 0
#             for i in range(length.numel()):
#                 l = length[i]
#                 texts.append(
#                     self.decode(
#                         t[index:index + l], torch.IntTensor([l]), raw=raw))
#                 index += l
#             return texts



# class strLabelConverterForCTC(object):
#     """Convert between str and label.
#
#     NOTE:
#         Insert `blank` to the alphabet for CTC.
#
#     Args:
#         alphabet (str): set of the possible characters.
#         ignore_case (bool, default=True): whether or not to ignore all of the case.
#     """
#
#     def __init__(self, alphabet, sep):
#         self.sep = sep
#         self.alphabet = alphabet.split(sep)
#         self.alphabet.append('-')  # for `-1` index
#
#         self.dict = {}
#         for i, item in enumerate(self.alphabet):
#             # NOTE: 0 is reserved for 'blank' required by wrap_ctc
#             self.dict[item] = i + 1
#
#     def encode(self, text):
#         """Support batch or single str.
#
#         Args:
#             text (str or list of str): texts to convert.
#
#         Returns:
#             torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
#             torch.IntTensor [n]: length of each text.
#         """
#         if isinstance(text, str):
#             text = text.split(self.sep)
#             text = [self.dict[item] for item in text]
#             length = [len(text)]
#         elif isinstance(text, collections.Iterable):
#             length = [len(s.split(self.sep)) for s in text]
#             text = self.sep.join(text)
#             text, _ = self.encode(text)
#         return (torch.IntTensor(text), torch.IntTensor(length))
#
#     def decode(self, t, length, raw=False):
#         """Decode encoded texts back into strs.
#
#         Args:
#             torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
#             torch.IntTensor [n]: length of each text.
#
#         Raises:
#             AssertionError: when the texts and its length does not match.
#
#         Returns:
#             text (str or list of str): texts to convert.
#         """
#         if length.numel() == 1:
#             length = length[0]
#             assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
#             if raw:
#                 return ''.join([self.alphabet[i - 1] for i in t])
#             else:
#                 char_list = []
#                 for i in range(length):
#                     if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
#                         char_list.append(self.alphabet[t[i] - 1])
#                 return ''.join(char_list)
#         else:
#             # batch mode
#             assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
#             texts = []
#             index = 0
#             for i in range(length.numel()):
#                 l = length[i]
#                 texts.append(
#                     self.decode(
#                         t[index:index + l], torch.IntTensor([l]), raw=raw))
#                 index += l
#             return texts


# class averager(object):
#     """Compute average for `torch.Variable` and `torch.Tensor`. """
#
#     def __init__(self):
#         self.reset()
#
#     def add(self, v):
#         if isinstance(v, Variable):
#             count = v.data.numel()
#             v = v.data.sum()
#         elif isinstance(v, torch.Tensor):
#             count = v.numel()
#             v = v.sum()
#
#         self.n_count += count
#         self.sum += v
#
#     def reset(self):
#         self.n_count = 0
#         self.sum = 0
#
#     def val(self):
#         res = 0
#         if self.n_count != 0:
#             res = self.sum / float(self.n_count)
#         return res


# def oneHot(v, v_length, nc):
#     batchSize = v_length.size(0)
#     maxLength = v_length.max()
#     v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
#     acc = 0
#     for i in range(batchSize):
#         length = v_length[i]
#         label = v[acc:acc + length].view(-1, 1).long()
#         v_onehot[i, :length].scatter_(1, label, 1.0)
#         acc += length
#     return v_onehot


# def loadData(v, data):
#     v.data.resize_(data.size()).copy_(data)


# def prettyPrint(v):
#     print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
#     print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
#                                               v.mean().data[0]))


# def assureRatio(img):
#     """Ensure imgH <= imgW."""
#     b, c, h, w = img.size()
#     if h > w:
#         main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
#         img = main(img)
#     return img


# class halo():
#     '''
#     u:高斯分布的均值
#     sigma:方差
#     nums:在一张图片中随机添加几个光点
#     prob:使用halo的概率
#     '''
#
#     def __init__(self, nums, u=0, sigma=0.2, prob=0.5):
#         self.u = u  # 均值μ
#         self.sig = math.sqrt(sigma)  # 标准差δ
#         self.nums = nums
#         self.prob = prob
#
#     def create_kernel(self, maxh=32, maxw=50):
#         height_scope = [10, maxh]  # 高度范围     随机生成高斯
#         weight_scope = [20, maxw]  # 宽度范围
#
#         x = np.linspace(self.u - 3 * self.sig, self.u + 3 * self.sig, random.randint(*height_scope))
#         y = np.linspace(self.u - 3 * self.sig, self.u + 3 * self.sig, random.randint(*weight_scope))
#         Gauss_map = np.zeros((len(x), len(y)))
#         for i in range(len(x)):
#             for j in range(len(y)):
#                 Gauss_map[i, j] = np.exp(-((x[i] - self.u) ** 2 + (y[j] - self.u) ** 2) / (2 * self.sig ** 2)) / (
#                         math.sqrt(2 * math.pi) * self.sig)
#
#         return Gauss_map
#
#     def __call__(self, img):
#         if random.random() < self.prob:
#             Gauss_map = self.create_kernel(32, 60)  # 初始化一个高斯核,32为高度方向的最大值，60为w方向
#             img1 = np.asarray(img)
#             img1.flags.writeable = True  # 将数组改为读写模式
#             nums = random.randint(1, self.nums)  # 随机生成nums个光点
#             img1 = img1.astype(np.float)
#             # print(nums)
#             for i in range(nums):
#                 img_h, img_w = img1.shape
#                 pointx = random.randint(0, img_h - 10)  # 在原图中随机找一个点
#                 pointy = random.randint(0, img_w - 10)
#
#                 h, w = Gauss_map.shape  # 判断是否超限
#                 endx = pointx + h
#                 endy = pointy + w
#
#                 if pointx + h > img_h:
#                     endx = img_h
#                     Gauss_map = Gauss_map[1:img_h - pointx + 1, :]
#                 if img_w < pointy + w:
#                     endy = img_w
#                     Gauss_map = Gauss_map[:, 1:img_w - pointy + 1]
#
#                 # 加上不均匀光照
#                 img1[pointx:endx, pointy:endy] = img1[pointx:endx, pointy:endy] + Gauss_map * 255.0
#             img1[img1 > 255.0] = 255.0  # 进行限幅，不然uint8会从0开始重新计数
#             img = img1
#         return Image.fromarray(np.uint8(img))


# class MyGaussianBlur(ImageFilter.Filter):
#     name = "GaussianBlur"
#
#     def __init__(self, radius=2, bounds=None):
#         self.radius = radius
#         self.bounds = bounds
#
#     def filter(self, image):
#         if self.bounds:
#             clips = image.crop(self.bounds).gaussian_blur(self.radius)
#             image.paste(clips, self.bounds)
#             return image
#         else:
#             return image.gaussian_blur(self.radius)


# class GBlur(object):
#     def __init__(self, radius=2, prob=0.5):
#         radius = random.randint(0, radius)
#         self.blur = MyGaussianBlur(radius=radius)
#         self.prob = prob
#
#     def __call__(self, img):
#         if random.random() < self.prob:
#             img = img.filter(self.blur)
#         return img


# class RandomBrightness(object):
#     """随机改变亮度
#         pil:pil格式的图片
#     """
#
#     def __init__(self, prob=1.5):
#         self.prob = prob
#
#     def __call__(self, pil):
#         rgb = np.asarray(pil)
#         if random.random() < self.prob:
#             hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
#             h, s, v = cv2.split(hsv)
#             adjust = random.choice([0.5, 0.7, 0.9, 1.2, 1.5, 1.7])  # 随机选择一个
#             # adjust = random.choice([1.2, 1.5, 1.7, 2.0])      # 随机选择一个
#             v = v * adjust
#             v = np.clip(v, 0, 255).astype(hsv.dtype)
#             hsv = cv2.merge((h, s, v))
#             rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
#         return Image.fromarray(np.uint8(rgb)).convert('L')


# class randapply(object):
#     """随机决定是否应用光晕、模糊或者二者都用
#
#     Args:
#         transforms (list or tuple): list of transformations
#     """
#
#     def __init__(self, transforms):
#         assert isinstance(transforms, (list, tuple))
#         self.transforms = transforms
#
#     def __call__(self, img):
#         for t in self.transforms:
#             img = t(img)
#         return img
#
#     def __repr__(self):
#         format_string = self.__class__.__name__ + '('
#         format_string += '\n    p={}'.format(self.p)
#         for t in self.transforms:
#             format_string += '\n'
#             format_string += '    {0}'.format(t)
#         format_string += '\n)'
#         return format_string


# def process_crnn(im_data, gtso, lbso, net, ctc_loss, converter, training):
#     num_gt = len(gtso)
#     rrois = []
#     labels = []
#     for kk in range(num_gt):
#         gts = gtso[kk]
#         lbs = lbso[kk]
#         if len(gts) != 0:
#             gt = np.asarray(gts)
#             center = (gt[:, 0, :] + gt[:, 1, :] + gt[:, 2, :] + gt[:, 3, :]) / 4        # 求中心点
#             dw = gt[:, 2, :] - gt[:, 1, :]
#             dh =  gt[:, 1, :] - gt[:, 0, :]
#             poww = pow(dw, 2)
#             powh = pow(dh, 2)
#             w = np.sqrt(poww[:, 0] + poww[:,1])
#             h = np.sqrt(powh[:,0] + powh[:,1])  + random.randint(-2, 2)
#             angle_gt = ( np.arctan2((gt[:,2,1] - gt[:,1,1]), gt[:,2,0] - gt[:,1,0]) + np.arctan2((gt[:,3,1] - gt[:,0,1]), gt[:,3,0] - gt[:,0,0]) ) / 2        # 求角度
#             angle_gt = -angle_gt / 3.1415926535 * 180                                   # 需要加个负号
#
#             # 10. 对每个rroi进行判断是否用于训练
#             for gt_id in range(0, len(gts)):
#
#                 gt_txt = lbs[gt_id]                       # 文字判断
#                 if gt_txt.startswith('##'):
#                     continue
#
#                 gt = gts[gt_id]                           # 标注信息判断
#                 if gt[:, 0].max() > im_data.size(3) or gt[:, 1].max() > im_data.size(2) or gt.min() < 0:
#                     continue
#
#                 rrois.append([kk, center[gt_id][0], center[gt_id][1], h[gt_id], w[gt_id], angle_gt[gt_id]])   # 将标注的rroi写入
#                 labels.append(gt_txt)
#
#     text, label_length = converter.encode(labels)
#
#     # 13.rroi_align, 特征前向传播，并求ctcloss
#     rois = torch.tensor(rrois).to(torch.float).cuda()
#     pooled_height = 32
#     maxratio = rois[:, 4] / rois[:, 3]
#     maxratio = maxratio.max().item()
#     pooled_width = math.ceil(pooled_height * maxratio)
#
#     roipool = _RRoiAlign(pooled_height, pooled_width, 1.0)  # 声明类
#     pooled_feat = roipool(im_data, rois.view(-1, 6))
#
#     # 13.1 显示所有的crop区域
#     alldebug = 0
#     if alldebug:
#         for i in range(pooled_feat.shape[0]):
#
#             x_d = pooled_feat.data.cpu().numpy()[i]
#             x_data_draw = x_d.swapaxes(0, 2)
#             x_data_draw = x_data_draw.swapaxes(0, 1)
#
#             x_data_draw += 1
#             x_data_draw *= 128
#             x_data_draw = np.asarray(x_data_draw, dtype=np.uint8)
#             x_data_draw = x_data_draw[:, :, ::-1]
#             cv2.imshow('crop %d' % i, x_data_draw)
#             cv2.imwrite('./data/tshow/crop%d.jpg' % i, x_data_draw)
#             # cv2.imwrite('./data/tshow/%s.jpg' % labels[i], x_data_draw)
#
#         for j in range(im_data.size(0)):
#             img = im_data[j].cpu().numpy().transpose(1,2,0)
#             img = (img + 1) * 128
#             img = np.asarray(img, dtype=np.uint8)
#             img = img[:, :, ::-1]
#             cv2.imshow('img%d'%j, img)
#             cv2.imwrite('./data/tshow/img%d.jpg' % j, img)
#         cv2.waitKey(100)
#
#     if training:
#         preds = net.ocr_forward(pooled_feat)
#
#         preds_size = Variable(torch.IntTensor([preds.size(0)] * preds.size(1)))       # 求ctc loss
#         res = ctc_loss(preds, text, preds_size, label_length) / preds.size(1)    # 求一个平均
#     else:
#         labels_pred = net.ocr_forward(pooled_feat)
#
#         _, labels_pred = labels_pred.max(2)
#         labels_pred = labels_pred.contiguous().view(-1)
#         # labels_pred = labels_pred.transpose(1, 0).contiguous().view(-1)
#         preds_size = Variable(torch.IntTensor([labels_pred.size(0)]))
#         res = converter.decode(labels_pred.data, preds_size.data, raw=False)
#         res = (res, labels)
#     return res


# class ImgDataset(Dataset):
#     def __init__(self, root=None, csv_root=None, transform=None, target_transform=None):
#         self.root = root
#         with open(csv_root) as f:
#             self.data = f.readlines()
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         per_label = self.data[idx].rstrip().split('\t')
#         imgpath = os.path.join(self.root, per_label[0])
#         srcimg = cv2.imread(imgpath)
#         img = srcimg[:, :, ::-1].copy()
#
#         if self.transform:
#             img = self.transform(img)
#             # img = torch.tensor(img, dtype=torch.float)
#         img = torch.from_numpy(img)
#         img = img.permute(2,0,1)
#         img = img.float()
#
#         temp = [[int(x) for x in per_label[2:6]]]
#
#         roi = []
#         for i in range(len(temp)):
#             temp1 = np.asarray([[temp[i][0], temp[i][3]], [temp[i][0],temp[i][1]], [temp[i][2],temp[i][1]], [temp[i][2],temp[i][3]]])
#             roi.append(temp1)
#
#         # for debug show
#         #     cv2.rectangle(srcimg, (temp1[1][0], temp1[1][1]), (temp1[3][0], temp1[3][1]), (255, 0, 0), thickness=2)
#         # #     temp1 = temp1.reshape(-1,1,2)
#         # #     cv2.polylines(srcimg,[temp1],False,(0,255,255), thickness=3)
#         # plt.imshow(srcimg)
#         # plt.show()
#
#         text = [per_label[1].lstrip(), per_label[6].lstrip()]
#
#
#         return img, roi, text          # gt_box的标注信息为x1,y1,x2,y2, 返回一个名字


# class ImgDataset2(Dataset):
#     def __init__(self, root=None, csv_root=None, transform=None, target_transform=None):
#         self.root = root
#         with open(csv_root) as f:
#             self.data = f.readlines()
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         per_label = self.data[idx].rstrip().split('\t')
#         imgpath = os.path.join(self.root, per_label[0])
#         srcimg = cv2.imread(imgpath)
#         img = srcimg[:, :, ::-1].copy()
#
#         if self.transform:
#             img = self.transform(img)
#             # img = torch.tensor(img, dtype=torch.float)
#         img = torch.from_numpy(img)
#         img = img.permute(2, 0, 1)
#         img = img.float()
#
#         temp = [[int(x) for x in per_label[2:6]],
#                 [int(x) for x in per_label[7:11]]]
#
#         roi = []
#         for i in range(len(temp)):
#             temp1 = np.asarray([[temp[i][0], temp[i][3]], [temp[i][0], temp[i][1]], [temp[i][2], temp[i][1]],
#                                 [temp[i][2], temp[i][3]]])
#             roi.append(temp1)
#
#         # cv2.rectangle(srcimg, (temp[0][0], temp[0][1]), (temp[0][2], temp[0][3]), (255, 0, 0), thickness=2)
#         #     temp1 = temp1.reshape(-1,1,2)
#         #     cv2.polylines(srcimg,[temp1],False,(0,255,255), thickness=3)
#         # plt.imshow(srcimg)
#         # plt.show()
#
#         text = [per_label[1].lstrip(), per_label[6].lstrip()]
#
#         return img, roi, text  # gt_box的标注信息为x1,y1,x2,y2, 返回一个名字


# def own_collate(batch):
#     r"""Puts each data field into a tensor with outer dimension batch size"""
#
#     error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
#     elem_type = type(batch[0])
#     img = []
#     gt_boxes = []
#     texts = []
#     for per_batch in batch:
#         img.append(per_batch[0])
#         gt_boxes.append(per_batch[1])
#         texts.append(per_batch[2])
#
#     return torch.stack(img, 0), gt_boxes, texts




class ImgAugTransform:
    def __init__(self):
        self.aug = A.Compose([
        A.CoarseDropout(max_holes=7, min_holes=1, min_height=1, min_width=1, max_height=16, max_width=4, fill_value=128,
                        p=0.5),
        A.OneOf([
            A.Blur(blur_limit=10,p=1),
            A.MedianBlur(blur_limit=5,p=1),

        ],)

        # A.RandomContrast(limit=0.05, p=0.75),
        # A.RandomBrightness(limit=0.05, p=0.75),
        # A.RandomBrightnessContrast(contrast_limit=0.05, brightness_limit=0.05, p=0.75),
    ])


    def __call__(self, img):
        # img = np.array(img)
        img = np.asarray(img, dtype=np.float32)
        img /= 128
        img -= 1

        transformed_img =  self.aug(image=img)['image']

        # return Image.fromarray(transformed_img)
        return transformed_img


def random_dilate(img):
    img = np.array(img)
    img = cv2.dilate(img, np.ones(shape=(random.randint(1, 3), random.randint(1, 3)), dtype=np.uint8))
    return Image.fromarray(img)
    # return img


def random_erode(img):
    img = np.array(img)
    img = cv2.erode(img, np.ones(shape=(random.randint(1, 3), random.randint(1, 3)), dtype=np.uint8))
    return Image.fromarray(img)
    # return img

def train_transforms():
    transform = transforms.Compose([

        transforms.RandomApply(
            [
                random_dilate,
            ],
            p=0.15),

        transforms.RandomApply(
            [
                random_erode,
            ],
            p=0.15),

        transforms.RandomAffine(degrees=3, scale=(0.95, 1.05), shear=3, resample=Image.NEAREST, fillcolor=255),
        transforms.RandomApply(
            [
                ImgAugTransform(),

            ],
            p=0.3),
        # transforms.ToTensor()

    ])



    return transform

def test_transforms():
  transform = transforms.Compose([
    transforms.ToTensor()
  ])
  return transform


trans_train = train_transforms()
test_train = test_transforms()


def cut_image(img, new_size, word_gto):

  if len(word_gto) > 0:
    rep = True
    cnt = 0
    while rep:

      if cnt > 30:
        return img

      text_poly = word_gto[random.randint(0, len(word_gto) - 1)]

      center = text_poly.sum(0) / 4

      xs = int(center[0] - random.uniform(-100, 100) - new_size[1] / 2)
      xs = max(xs, 1)
      ys = int(center[1] - random.uniform(-100, 100) - new_size[0] / 2)
      ys = max(ys, 1)

      crop_rect = (xs, ys, xs + new_size[1], ys + new_size[0])
      crop_img = img[crop_rect[1]:crop_rect[3], crop_rect[0]:crop_rect[2]]
      # cv2.imshow('dasd',crop_img)

      if crop_img.shape[0] == crop_img.shape[1]:
        rep = False
      else:
        cnt += 1


  else:
    xs = int(random.uniform(0, img.shape[1]))
    ys = int(random.uniform(0, img.shape[0]))
    crop_rect = (xs, ys, xs + new_size[1], ys + new_size[0])
    crop_img = img[crop_rect[1]:crop_rect[3], crop_rect[0]:crop_rect[2]]

  if len(word_gto) > 0:
    word_gto[:, :, 0] -= xs
    word_gto[:, :, 1] -= ys

  return crop_img

class ocrDataset(Dataset):
    def __init__(self, root, norm_height = 48,in_train = True, target_transform=None):
        self.norm_height = norm_height
        self.path = self.get_path(root)
        self.root = root
        self.train_transform = trans_train
        self.test_transform = test_train
        self.in_train = in_train
        self.target_transform = target_transform

    def get_path(self,data_path):
        base_dir = os.path.dirname(data_path)
        files_out = []
        cnt = 0
        with open(data_path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) == 0:
                    continue
                if not line[0] == '/':
                    line = '{0}/{1}'.format(base_dir, line)
                files_out.append(line)
                cnt += 1
                # if cnt > 100:
                #  break
        return files_out
    def get_data(self,image_name):
        src_del = " "
        spl = image_name.split(" ")
        if len(spl) == 1:
            spl = image_name.split(",")
            src_del = ","
        image_name = spl[0].strip()
        # image_name = (spl[0] + ' '+ spl[1]).strip()
        gt_txt = ''
        if len(spl) > 1:
            gt_txt = ""
            delim = ""
            for k in range(1, len(spl)):
                gt_txt += delim + spl[k]
                delim = src_del
            if len(gt_txt) > 1 and gt_txt[0] == '"' and gt_txt[-1] == '"':
                gt_txt = gt_txt[1:-1]


        if image_name[len(image_name) - 1] == ',':
            image_name = image_name[0:-1]

        im = cv2.imread(image_name)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im[2:im.shape[0]-2,:,:]
        scale = self.norm_height / float(im.shape[0])
        width = int(im.shape[1] * scale)

        im = cv2.resize(im, (int(width), self.norm_height))
        image = PIL.Image.fromarray(np.uint8(im))
        # image = np.asarray(im, dtype=np.float32)
        # image /= 128
        # image -= 1
#get labels
        # gt_labels = []
        # for k in range(len(gt_txt)):
        #     if gt_txt[k] in self.codec_rev:
        #         gt_labels.append(self.codec_rev[gt_txt[k]])
        #     else:
        #         gt_labels.append(3)
        #
        # return image,gt_labels
        return image,gt_txt

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        try:
            image,label = self.get_data(self.path[index])
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]
        if self.in_train :
            image = self.train_transform(image)
        else:
            image = self.test_transform(image)
            # image = np.array(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return (image, label)


class alignCollate(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        images, labels = zip(*batch)
        c = images[0].size(0)
        h = max([p.size(1) for p in images])
        w = max([p.size(2) for p in images])
        batch_images = torch.zeros(len(images), c, h, w).fill_(1)
        for i, image in enumerate(images):
            started_h = max(0, random.randint(0, h - image.size(1)))
            started_w = max(0, random.randint(0, w - image.size(2)))
            batch_images[i, :, started_h:started_h + image.size(1), started_w:started_w + image.size(2)] = image
        return batch_images, labels

    # Transform


class E2Edataset(Dataset):
    def __init__(self, train_list, input_size=512):
        super(E2Edataset, self).__init__()
        self.image_list = np.array(get_images(train_list))
        self.input_size = input_size

        print('{} training images in {}'.format(self.image_list.shape[0], train_list))

        self.transform = transforms.Compose([
                    transforms.ColorJitter(.3,.3,.3,.3),
                    transforms.RandomGrayscale(p=0.1)  ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        im_name = self.image_list[index]

        im = cv2.imread(im_name)                # 图片
        # cv2.imshow('sad',im)
        txt_fn = im_name.replace(os.path.basename(im_name).split('.')[1], 'txt')
        base_name = os.path.basename(txt_fn)
        txt_fn_gt = '{0}/gt_{1}'.format(os.path.dirname(im_name), base_name)

        text_polys, text_tags, labels_txt = load_gt_annoataion(txt_fn_gt, True)
        # 载入标注信息
        resize_w = self.input_size
        resize_h = self.input_size

        scaled = cut_image(im, (self.input_size, self.input_size), text_polys)
        if scaled.shape[0] == 0 or scaled.shape[1] == 0:
            scaled = im

        if scaled.shape[1] != resize_w or scaled.shape[0] != resize_h:
            ratio_img= min(scaled.shape[1]/scaled.shape[0],scaled.shape[0]/scaled.shape[1])
            if scaled.shape[0] > scaled.shape[1]:
                resize_w =resize_w * ratio_img
                scalex = scaled.shape[1] / resize_w
                scaley = scaled.shape[0] / resize_h
                scaled = cv2.resize(scaled, dsize=(int(resize_w), int(resize_h)))
            else :
                resize_h = resize_h * ratio_img
                scalex = scaled.shape[1] / resize_w
                scaley = scaled.shape[0] / resize_h
                scaled = cv2.resize(scaled, dsize=(int(resize_w), int(resize_h)))
            # continue

            if len(text_polys) > 0:
                text_polys[:, :, 0] /= scalex
                text_polys[:, :, 1] /= scaley
            scaled = cv2.copyMakeBorder(scaled, 0, self.input_size - scaled.shape[0], 0, self.input_size - scaled.shape[1], cv2.BORDER_CONSTANT,value=[255,255,255])


            # scaled_show = scaled.copy()
            # for (i, c) in enumerate(text_polys):
            #     draw_box_points(scaled_show, c, color=(0, 255, 0), thickness=1)
            #     # instead of creating a new image, I simply modify the old one
            #
            #
            # # show the modified image with all the rectangles at the end.
            # cv2.imshow('img', scaled_show)

        pim = PIL.Image.fromarray(np.uint8(scaled))
        if self.transform:
            pim = self.transform(pim)
        im = np.array(pim)


        new_h, new_w, _ = im.shape
        score_map, geo_map, training_mask, gt_idx, gt_out, labels_out = generate_rbox(im, (new_h, new_w), text_polys, text_tags, labels_txt, vis=False)

        im = np.asarray(im, dtype=np.float32)
        im /= 128
        im -= 1
        im = torch.from_numpy(im).permute(2,0,1)



        return im, score_map, geo_map, training_mask, gt_idx, gt_out, labels_out


def E2Ecollate(batch):
    img = []
    gt_boxes = []
    texts = []
    scores = []
    training_masks = []
    for per_batch in batch:
        img.append(per_batch[0])
        scores.append(per_batch[1])
        training_masks.append(per_batch[3])
        gt_boxes.append(per_batch[5])
        texts.append(per_batch[6])


    return torch.stack(img, 0), gt_boxes, texts,scores,training_masks



if __name__ == '__main__':
    llist = './data/ICDAR2015.txt'

    data = E2Edataset(train_list=llist)

    E2Edataloader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=False, collate_fn=E2Ecollate)

    for index, data in enumerate(E2Edataloader):
        im = data