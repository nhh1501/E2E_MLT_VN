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
import tensorflow as tf
from data_gen import draw_box_points


class ImgAugTransform:
    def __init__(self):
        self.aug = A.Compose([

        A.OneOf([
            A.Blur(blur_limit=10,p=0.5),
            A.MedianBlur(blur_limit=5,p=0.5),
        ]),
        A.OneOf([
            A.CoarseDropout(max_holes=7, min_holes=3, min_height=3, min_width=1, max_height=16, max_width=4,
                            fill_value=255,
                            p=0.5),
            A.CoarseDropout(max_holes=7, min_holes=3, min_height=3, min_width=1, max_height=16, max_width=4,
                            fill_value=170,
                            p=0.5),
            A.CoarseDropout(max_holes=7, min_holes=3, min_height=3, min_width=1, max_height=16, max_width=4,
                            fill_value=85,
                            p=0.5),
            A.CoarseDropout(max_holes=7, min_holes=3, min_height=3, min_width=1, max_height=16, max_width=4,
                            fill_value=0,
                        p=0.5),

        ]),

        # A.RandomContrast(limit=0.05, p=0.75),
        # A.RandomBrightness(limit=0.05, p=0.75),
        # A.RandomBrightnessContrast(contrast_limit=0.05, brightness_limit=0.05, p=0.75),
    ])


    def __call__(self, img):
        img = np.array(img)
        transformed_img =  self.aug(image=img)['image']

        return Image.fromarray(transformed_img)
        # return transformed_img


def to_array(img):
    img = np.asarray(img, dtype=np.float32)
    img /= 128
    img -= 1
    return img
    # return img

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

        transforms.RandomAffine(degrees=3, scale=(0.95, 1.05), shear=3, resample=Image.NEAREST, fillcolor=(255,255,255)),
        # transforms.RandomApply(
        #     [
        #         to_array,
        #     ],
        #     p=1),
        transforms.RandomApply(
            [
                ImgAugTransform(),

            ],
            p=0.6),
        transforms.ToTensor()

    ])



    return transform

def test_transforms():
  transform = transforms.Compose([
    transforms.ToTensor()
  ])
  return transform


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
        self.train_transform = train_transforms()
        self.test_transform = test_transforms()
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
    def __init__(self, train_list, input_size=512, normalize = True):
        super(E2Edataset, self).__init__()
        self.image_list = np.array(get_images(train_list))
        self.input_size = input_size
        self.normalize = normalize

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


        im = np.asarray(im)
        ##
        if self.normalize:
            im = im.astype(np.float32)
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


class BeamSearchDecoder():
    def __init__(self, lib, corpus, chars, word_chars, beam_width=20, lm_type='Words', lm_smoothing=0.01, tfsess=None):
        word_beam_search_module = tf.load_op_library(lib)
        self.mat = tf.placeholder(tf.float32)
        corpus = open(corpus).read()
        chars = open(chars).read()
        word_chars = open(word_chars).read()

        self.beamsearch_decoder = word_beam_search_module.word_beam_search(self.mat, beam_width, lm_type, lm_smoothing,
                                                                           corpus, chars, word_chars)
        self.tfsess = tfsess or tf.Session()
        self.idx2char = dict(zip(range(0, len(chars)), chars))

    def beamsearch(self, mat):
        mat = np.concatenate((mat[:, :, 1:], mat[:, :, :1]), axis=-1)
        results = self.tfsess.run(self.beamsearch_decoder, {self.mat: mat})
        return results

    def decode(self, preds_idx):
        return [''.join([self.idx2char[idx] for idx in row if idx < len(self.idx2char)]) for row in preds_idx]


if __name__ == '__main__':
    llist = './data/ICDAR2015.txt'

    data = E2Edataset(train_list=llist)

    E2Edataloader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=False, collate_fn=E2Ecollate)

    for index, data in enumerate(E2Edataloader):
        im = data