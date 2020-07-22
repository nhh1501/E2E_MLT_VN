'''
Edit form train_ocr
'''
import os, sys
sys.path.append('./build')
device = 'cuda'
import numpy as np
import torch.nn.functional as F
import collections
import net_utils
import editdistance as ed
from ocr_utils import ocr_image, ocr_batch, crnn_batch
import torch
f = open('codec.txt', 'r')
codec = f.readlines()[0]
codec_rev = {}
index = 4
for i in range(0, len(codec)):
  codec_rev[codec[i]] = index
  index += 1
f.close()

def cer(predict, gt):
  distance = ed.eval(predict, gt)
  return distance,len(gt)

class strLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
      self._ignore_case = ignore_case
      if self._ignore_case:
        alphabet = alphabet.lower()
      self.alphabet = alphabet + '-'  # for `-1` index

      self.dict = {}
      index = 4
      for char in (alphabet):
        # NOTE: 0 is reserved for 'blank' required by wrap_ctc
        self.dict[char] = index
        index += 1

    def encode(self, text):
      """Support batch or single str.
      Args:
          text (str or list of str): texts to convert.
      Returns:
          torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
          torch.IntTensor [n]: length of each text.
      """
      if isinstance(text, str):
        texts = []
        for char in text:
          if char in self.dict:
            texts.append(self.dict[char.lower() if self._ignore_case else char])
          else:
            texts.append(3)

        length = [len(text)]
      elif isinstance(text, collections.Iterable):
        length = [len(s) for s in text]
        text = ''.join(text)
        texts, _ = self.encode(text)

      return (torch.IntTensor(texts), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
      """Decode encoded texts back into strs.
      Args:
          torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
          torch.IntTensor [n]: length of each text.
      Raises:
          AssertionError: when the texts and its length does not match.
      Returns:
          text (str or list of str): texts to convert.
      """
      if length.numel() == 1:
        length = length[0]
        assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                     length)
        if raw:
          return ''.join([self.alphabet[i - 4] for i in t])
        else:
          char_list = []
          for i in range(length):
            if t[i] > 3 and t[i] < (len(self.alphabet) + 4) and (not (i > 0 and t[i - 1] == t[i])):
              char_list.append(self.alphabet[t[i] - 4])
            elif t[i] == 3 and (not (i > 0 and t[i - 1] == t[i])):
              char_list.append(' ')
          return ''.join(char_list)
      else:
        # batch mode
        assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
          t.numel(), length.sum())
        texts = []
        index = 0
        for i in range(length.numel()):
          l = length[i]
          texts.append(
            self.decode(
              t[index:index + l], torch.IntTensor([l]), raw=raw))
          index += l
        return texts

def dice_loss(segm_preds, score_maps,training_masks,multi_scale = False):

  score_maps = np.asarray(score_maps, dtype=np.uint8)
  training_masks = np.asarray(training_masks, dtype=np.uint8)

  smaps_var = net_utils.np_to_variable(score_maps, is_cuda=False)
  training_mask_var = net_utils.np_to_variable(training_masks, is_cuda=False)
  segm_pred = segm_preds[0].squeeze(1)
  segm_pred1 = segm_preds[1].squeeze(1)
  inp = segm_pred * training_mask_var
  target = smaps_var * training_mask_var

  smooth = 1.
  iflat = inp.view(-1)
  tflat = target.view(-1)
  intersection = (iflat * tflat).sum()
  result = - ((2. * intersection + smooth) /
            (iflat.sum() + tflat.sum() + smooth))
  if multi_scale:
    iou_gts = F.interpolate(smaps_var.unsqueeze(1), size=(segm_pred1.size(1), segm_pred1.size(2)), mode='bilinear',
                            align_corners=True).squeeze(1)
    iou_masks = F.interpolate(training_mask_var.unsqueeze(1), size=(segm_pred1.size(1), segm_pred1.size(2)), mode='bilinear',
                              align_corners=True).squeeze(1)
    inp2 = segm_pred1 * iou_masks
    target2 = iou_gts * iou_masks

    # smooth = 1.
    iflat2 = inp2.view(-1)
    tflat2 = target2.view(-1)
    intersection2 = (iflat2 * tflat2).sum()
    result += - ((2. * intersection2 + smooth) /
            (iflat2.sum() + tflat2.sum() + smooth))

  return result

def evaluate(e2edataloader,net):
  net.eval()
  num_count = 0
  norm_height = 44
  distance_sum = 0
  len_cer = 0
  loss_seg = 0
  len_wer = 0
  with torch.no_grad():
    for index, date in enumerate(e2edataloader):
      im_data, gtso, lbso, score_maps, training_masks = date
      im_data = im_data.to(device)
        #get_loss _ cer _ wer
      res = ocr_batch(net, codec, im_data, gtso, lbso, norm_height)
      pred, target = res
      if not isinstance(pred, list):
        pred = [pred]
      assert (len(pred) == len(target))

      target_cer = ''.join(target).replace(" ", "").lower()
      pred_cer = ''.join(pred).replace(" ", "").lower()
      distance,len_sum= cer(pred_cer,target_cer)
      distance_sum += distance
      len_cer += len_sum
      len_wer += len(pred)
      for idx in range (len(pred)):
        if pred[idx].replace(" ", "").lower() == target[idx].replace(" ", "").lower():
          num_count += 1
    WER = 1 - (num_count / len_wer)
    CER = distance_sum / len_cer
    return CER, WER

def evaluate_crnn(e2edataloader,net):
  #load image form datagen _ eval end2end
  net.eval()
  norm_height = 48
  num_count = 0
  distance_sum = 0
  len_cer = 0
  len_wer = 0
  with torch.no_grad():
    for index, date in enumerate(e2edataloader):
      im_data, gtso, lbso, score_maps, training_masks = date
      im_data = im_data.to(device)
        #get_loss _ cer _ wer
      # res = ocr_batch(net, codec, im_data, ctc_loss, gtso, lbso)
      res = crnn_batch(net, codec, im_data, gtso, lbso, norm_height)

      pred, target = res
      if not isinstance(pred, list):
        pred = [pred]
      assert (len(pred) == len(target))

      target_cer = ''.join(target).replace(" ", "").lower()
      pred_cer = ''.join(pred).replace(" ", "").lower()
      distance,len_sum= cer(pred_cer,target_cer)
      distance_sum += distance
      len_cer += len_sum
      len_wer += len(pred)
      for idx in range (len(pred)):
        if pred[idx].replace(" ", "").lower() == target[idx].replace(" ", "").lower():
          num_count += 1
    WER = 1 - (num_count / len_wer)
    CER = distance_sum / len_cer
    return CER, WER

def eval_ocr(ocrdataloader,net):
  # norm_height = 44
  net.eval()
  converter = strLabelConverter(codec)
  num_count = 0
  distance_sum = 0
  len_cer = 0
  len_wer = 0
  with torch.no_grad():
    for index, date in enumerate(ocrdataloader):
      print(index)
      im_data, lbso = date
      im_data = im_data.to(device)
      features = net.forward_features(im_data)
      labels_pred = net.forward_ocr(features)

      ctc_f = labels_pred.data.cpu().numpy()
      ctc_f = ctc_f.swapaxes(1, 2)
      labelss = ctc_f.argmax(2)
      leng=[labelss.shape[1] for i in range(labelss.shape[0])]

      labelss.resize((1, labelss.shape[0] * labelss.shape[1]))
      pred = converter.decode(torch.IntTensor(labelss[0,:]),torch.IntTensor(leng))
      target = lbso

      if not isinstance(pred, list):
        pred = [pred]
      assert (pred.__len__() == target.__len__())

      target_cer = ''.join(target).replace(" ", "").lower()
      pred_cer = ''.join(pred).replace(" ", "").lower()
      distance,len_sum= cer(pred_cer,target_cer)
      distance_sum += distance
      len_cer += len_sum
      len_wer += len(target)
      for idx in range (len(target)):
        if pred[idx].replace(" ", "").lower() == target[idx].replace(" ", "").lower():
          num_count += 1
    WER = 1 - (num_count / len_wer)
    CER = distance_sum / len_cer
    return CER, WER

def eval_ocr_crnn(ocrdataloader,net):
  # norm_height = 48
  net.eval()
  converter = strLabelConverter(codec)
  num_count = 0
  distance_sum = 0
  len_cer = 0
  len_wer = 0
  with torch.no_grad():
    for index, date in enumerate(ocrdataloader):
      print(index)
      im_data, lbso = date
      im_data = im_data.to(device)
      labels_pred = net.forward_ocr(im_data)

      ctc_f = labels_pred.data.cpu().numpy()
      ctc_f = ctc_f.swapaxes(1, 2)
      labelss = ctc_f.argmax(2)
      leng=[labelss.shape[1] for i in range(labelss.shape[0])]

      labelss.resize((1, labelss.shape[0] * labelss.shape[1]))
      pred = converter.decode(torch.IntTensor(labelss[0,:]),torch.IntTensor(leng))
      target = lbso

      if not isinstance(pred, list):
        pred = [pred]
      assert (pred.__len__() == target.__len__())

      target_cer = ''.join(target).replace(" ", "").lower()
      pred_cer = ''.join(pred).replace(" ", "").lower()
      distance,len_sum= cer(pred_cer,target_cer)
      distance_sum += distance
      len_cer += len_sum
      len_wer += len(target)
      for idx in range (len(target)):
        if pred[idx].replace(" ", "").lower() == target[idx].replace(" ", "").lower():
          num_count += 1
    WER = 1 - (num_count / len_wer)
    CER = distance_sum / len_cer
    return CER, WER




