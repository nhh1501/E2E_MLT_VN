'''
Edit form train_ocr
'''
import os, sys
import numpy as np
import cv2
import net_utils
import math
import torch.nn.functional as F
import collections
import glob
import csv
import editdistance
import torch
from ocr_utils import ocr_image, ocr_batch, crnn_batch    # next(iter(data_loader))
from data_gen import draw_box_points
from ocr_utils import print_seq_ext
from demo import resize_image
from nms import get_boxes
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
sys.path.append('./build')


f = open('codec.txt', 'r')
codec = f.readlines()[0]
codec_rev = {}
index = 4
for i in range(0, len(codec)):
  codec_rev[codec[i]] = index
  index += 1
f.close()
device = 'cuda'



def draw_text_points(img, det_text, box, color=(255, 255, 255)):
  # "draw text to img"
  font = ImageFont.truetype("Arial-Unicode-Regular.ttf", 25)
  img = Image.fromarray(img)
  center = (box[0, :] + box[1, :] + box[2, :] + box[3, :]) / 4
  pil_draw = ImageDraw.Draw(img)
  pil_draw.text(center, det_text, fill=color, font=font)
  return np.array(img)

def load_gt(p, is_icdar=False):
  '''
  load annotation from the text file,
  :param p:
  :return:
  '''
  text_polys = []
  text_gts = []
  if not os.path.exists(p):
    return np.array(text_polys, dtype=np.float32), text_gts
  with open(p, 'r') as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')
    for line in reader:
      # strip BOM. \ufeff for python3,	\xef\xbb\bf for python2
      line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

      x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
      # cls = 0
      gt_txt = ''
      delim = ''
      start_idx = 8
      if is_icdar:
        start_idx = 8

      for idx in range(start_idx, len(line)):
        gt_txt += delim + line[idx]
        delim = ','

      text_polys.append([x4, y4, x1, y1, x2, y2, x3, y3])
      text_line = gt_txt.strip()

      text_gts.append(text_line)

    return np.array(text_polys, dtype=np.float), text_gts

def intersect(a, b):
  '''Determine the intersection of two rectangles'''
  rect = (0, 0, 0, 0)
  r0 = max(a[0], b[0])
  c0 = max(a[1], b[1])
  r1 = min(a[2], b[2])
  c1 = min(a[3], b[3])
  # Do we have a valid intersection?
  if r1 > r0 and c1 > c0:
    rect = (r0, c0, r1, c1)
  return rect

def union(a, b):
  r0 = min(a[0], b[0])
  c0 = min(a[1], b[1])
  r1 = max(a[2], b[2])
  c1 = max(a[3], b[3])
  return (r0, c0, r1, c1)

def area(a):
  '''Computes rectangle area'''
  width = a[2] - a[0]
  height = a[3] - a[1]
  return abs(width * height)

def evaluate_image(img, detections, gt_rect, gt_txts, iou_th=0.5, iou_th_vis=0.5, iou_th_eval=0.5, eval_text_length=3):
  '''
  Summary : Returns end-to-end true-positives, detection true-positives, number of GT to be considered for eval (len > 2).
  Description : For each predicted bounding-box, comparision is made with each GT entry. Values of number of end-to-end true
                              positives, number of detection true positives, number of GT entries to be considered for evaluation are computed.

  Parameters
  ----------
  iou_th_eval : float
          Threshold value of intersection-over-union used for evaluation of predicted bounding-boxes
  iou_th_vis : float
          Threshold value of intersection-over-union used for visualization when transciption is true but IoU is lesser.
  iou_th : float
          Threshold value of intersection-over-union between GT and prediction.
  word_gto : list of lists
          List of ground-truth bounding boxes along with transcription.
  batch : list of lists
          List containing data (input image, image file name, ground truth).
  detections : tuple of tuples
          Tuple of predicted bounding boxes along with transcriptions and text/no-text score.

  Returns
  -------
  tp : int
          Number of predicted bounding-boxes having IoU with GT greater than iou_th_eval.
  tp_e2e : int
          Number of predicted bounding-boxes having same transciption as GT and len > 2.
  gt_e2e : int
          Number of GT entries for which transcription len > 2.
  '''

  gt_to_detection = {}
  detection_to_gt = {}
  tp = 0
  tp_e2e = 0
  tp_e2e_ed1 = 0
  gt_e2e = 0

  gt_matches = np.zeros(gt_rect.shape[0])
  gt_matches_ed1 = np.zeros(gt_rect.shape[0])

  # print('\n')
  for i in range(0, len(detections)):

    det = detections[i]
    box = det[0]  # Predicted bounding-box parameters
    box = np.array(box, dtype="int")  # Convert predicted bounding-box to numpy array
    box = box[0:8].reshape(4, 2)
    bbox = cv2.boundingRect(box)

    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
    bbox[2] += bbox[0]  # Convert width to right-coordinate
    bbox[3] += bbox[1]  # Convert height to bottom-coordinate

    det_text = det[1]  # Predicted transcription for bounding-box

    for gt_no in range(len(gt_rect)):

      gtbox = gt_rect[gt_no]
      txt = gt_txts[gt_no]  # GT transcription for given GT bounding-box
      gtbox = np.array(gtbox, dtype="int")
      gtbox = gtbox[0:8].reshape(4, 2)
      rect_gt = cv2.boundingRect(gtbox)

      rect_gt = [rect_gt[0], rect_gt[1], rect_gt[2], rect_gt[3]]
      rect_gt[2] += rect_gt[0]  # Convert GT width to right-coordinate
      rect_gt[3] += rect_gt[1]  # Convert GT height to bottom-coordinate

      inter = intersect(bbox, rect_gt)  # Intersection of predicted and GT bounding-boxes
      uni = union(bbox, rect_gt)  # Union of predicted and GT bounding-boxes
      ratio = area(inter) / float(area(uni))  # IoU measure between predicted and GT bounding-boxes

      # 1). Visualize the predicted-bounding box if IoU with GT is higher than IoU threshold (iou_th) (Always required)
      # 2). Visualize the predicted-bounding box if transcription matches the GT and condition 1. holds
      # 3). Visualize the predicted-bounding box if transcription matches and IoU with GT is less than iou_th_vis and 1. and 2. hold
      if ratio > iou_th:
        ###
        img = draw_text_points(img, det_text, box, color=(0, 255, 255))
        ###
        if not gt_no in gt_to_detection:
          gt_to_detection[gt_no] = [0, 0]

        edit_dist = editdistance.eval(det_text.lower(), txt.lower())
        #############
        # print('{0}___ratio:{1}___editdist:{2}'.format(det_text, ratio, edit_dist))
        ##
        # edit_dist =0 - draw GREEN
        # edit_dist > 0 - draw BLUE
        # not match IOU - draw RED

        if edit_dist <= 1:
          gt_matches_ed1[gt_no] = 1

        if edit_dist == 0:  # det_text.lower().find(txt.lower()) != -1:
          draw_box_points(img, box, color=(0, 255, 0), thickness=2)  # GREEN - edit_distant = 0
          gt_matches[gt_no] = 1  # Change this parameter to 1 when predicted transcription is correct.
          if ratio < iou_th_vis:
            # draw_box_points(draw, box, color = (255, 255, 255), thickness=2)
            # cv2.imshow('draw', draw)
            # cv2.waitKey(0)
            pass

        '''
        gt_to_dectection {gt_no :[ratio,idx_predict] }  - ground_true thứ gt_no trùng với predict thứ idx_predict
        detection_to_gt {id_predict :[gt_no,ratio,edit_dist]}  - predict thứ idx_predict trùng với ground_true thứ gt_no 
            #gt_no : ứng với dòng tứ gt_no trong file gt_txt
            #idx_predict : ứng với predict thứ idx_predict trong output list của model
            #radio : IOU
            #edit_dist : Character error rate (CER)

        '''

        tupl = gt_to_detection[gt_no]
        if tupl[0] < ratio:
          tupl[0] = ratio
          tupl[1] = i
          detection_to_gt[i] = [gt_no, ratio, edit_dist]

  # Count the number of end-to-end and detection true-positives
  ##
  # cv2.imshow('draw', img)
  # cv2.waitKey(0)

  # cv2.imwrite('preview/{0}'.format(base_nam), img)

  ##
  for gt_no in range(gt_matches.shape[0]):
    gt = gt_matches[gt_no]
    gt_ed1 = gt_matches_ed1[gt_no]
    txt = gt_txts[gt_no]

    gtbox = gt_rect[gt_no]
    gtbox = np.array(gtbox, dtype="int")
    gtbox = gtbox[0:8].reshape(4, 2)

    if len(txt) >= eval_text_length and not txt.startswith('##'):
      gt_e2e += 1
      if gt == 1:
        tp_e2e += 1
      if gt_ed1 == 1:
        tp_e2e_ed1 += 1

    if gt_no in gt_to_detection:
      tupl = gt_to_detection[gt_no]
      if tupl[0] > iou_th_eval:  # Increment detection true-positive, if IoU is greater than iou_th_eval
        if len(txt) >= eval_text_length and not txt.startswith('##'):
          tp += 1
      else:
        pass
      # draw_box_points(img, gtbox, color = (255, 255, 255), thickness=2)

  for i in range(0, len(detections)):
    det = detections[i]
    box = det[0]  # Predicted bounding-box parameters
    box = np.array(box, dtype="int")  # Convert predicted bounding-box to numpy array
    box = box[0:8].reshape(4, 2)

    if not i in detection_to_gt:
      draw_box_points(img, box, color=(0, 0, 255), thickness=2)  # RED - not match IOU > 0.5
      img = draw_text_points(img, det[1], box, color=(0, 0, 255))
    else:
      [gt_no, ratio, edit_dist] = detection_to_gt[i]
      if edit_dist > 0:
        draw_box_points(img, box, color=(255, 0, 0), thickness=2)  # BLUE - edit_distant > 0

  # cv2.imshow('draw', draw)
  # print('Missing:', len(detections) - len(detection_to_gt))
  return tp, tp_e2e, gt_e2e, tp_e2e_ed1, detection_to_gt, img

def cer(predict, gt):
  distance = editdistance.eval(predict, gt)
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
  '''
  Description : Eval E2E. Batch will be got --> Cut rec (x1y1,x2y2,x3y3,x4y4) --> predict
  :param e2edataloader:
  :param net:
  :return:
  '''
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
  '''
  Description: load valid_ocr to predict
  :param ocrdataloader:
  :param net: crnn
  :return: CER WER
  '''
  # norm_height = 44
  net.eval()
  converter = strLabelConverter(codec)
  num_count = 0
  distance_sum = 0
  len_cer = 0
  len_wer = 0
  with torch.no_grad():
    for index, date in enumerate(ocrdataloader):
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
      assert (len(pred) == len(target))

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
  # Decription : evaluate OCR model
  # norm_height = 48
  net.eval()
  converter = strLabelConverter(codec)
  num_count = 0
  distance_sum = 0
  len_cer = 0
  len_wer = 0
  with torch.no_grad():
    for index, date in enumerate(ocrdataloader):
      im_data, lbso = date
      im_data = im_data.to(device)
      labels_pred = net.forward_ocr(im_data)
      labels_pred = labels_pred.permute(1, 2, 0)
      ctc_f = labels_pred.data.cpu().numpy()
      ctc_f = ctc_f.swapaxes(1, 2)
      labelss = ctc_f.argmax(2)
      leng=[labelss.shape[1] for i in range(labelss.shape[0])]

      labelss.resize((1, labelss.shape[0] * labelss.shape[1]))
      pred = converter.decode(torch.IntTensor(labelss[0,:]),torch.IntTensor(leng))
      target = lbso

      if not isinstance(pred, list):
        pred = [pred]
      assert (len(pred)== len(target))

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

def evaluate_e2e(root, net, norm_height = 40,name_model='E2E', normalize= False ,save = False, cuda= True,save_dir = 'eval'):
  #Decription : evaluate model E2E
  net = net.eval()
  # if cuda:
  #   print('Using cuda ...')
  #   net = net.to(device)


  images = glob.glob(os.path.join(root, '*.jpg'))
  png = glob.glob(os.path.join(root, '*.png'))
  images.extend(png)
  png = glob.glob(os.path.join(root, '*.JPG'))
  images.extend(png)

  imagess = np.asarray(images)

  tp_all = 0
  gt_all = 0
  tp_e2e_all = 0
  gt_e2e_all = 0
  tp_e2e_ed1_all = 0
  detecitons_all = 0
  eval_text_length = 2
  segm_thresh = 0.5
  min_height = 8
  idx =0

  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  note_path = os.path.join(save_dir, 'note_eval.txt')
  note_file = open(note_path, 'a')


  with torch.no_grad():

    index = np.arange(0, imagess.shape[0])
    # np.random.shuffle(index)
    for i in index:
      img_name = imagess[i]
      base_nam = os.path.basename(img_name)
      #
      # if args.evaluate == 1:
      res_gt = base_nam.replace(".jpg", '.txt').replace(".png", '.txt')
      res_gt = '{0}/gt_{1}'.format(root, res_gt)
      if not os.path.exists(res_gt):
        res_gt = base_nam.replace(".jpg", '.txt').replace("_", "")
        res_gt = '{0}/gt_{1}'.format(root, res_gt)
        if not os.path.exists(res_gt):
          print('missing! {0}'.format(res_gt))
          gt_rect, gt_txts = [], []
      # continue
      gt_rect, gt_txts = load_gt(res_gt)


      # print(img_name)
      img = cv2.imread(img_name)

      im_resized,_ = resize_image(img, max_size=1848 * 1024, scale_up=False)  # 1348*1024 #1848*1024
      images = np.asarray([im_resized], dtype=np.float)

      if normalize:
        images /= 128
        images -= 1
      im_data = net_utils.np_to_variable(images, is_cuda=cuda).permute(0, 3, 1, 2)

      [iou_pred, iou_pred1], rboxs, angle_pred, features = net(im_data)
      iou = iou_pred.data.cpu()[0].numpy()
      iou = iou.squeeze(0)

      rbox = rboxs[0].data.cpu()[0].numpy()
      rbox = rbox.swapaxes(0, 1)
      rbox = rbox.swapaxes(1, 2)

      detections = get_boxes(iou, rbox, angle_pred[0].data.cpu()[0].numpy(), segm_thresh)

      im_scalex = im_resized.shape[1] / img.shape[1]
      im_scaley = im_resized.shape[0] / img.shape[0]

      detetcions_out = []
      detectionso = np.copy(detections)
      if len(detections) > 0:
        detections[:, 0] /= im_scalex
        detections[:, 2] /= im_scalex
        detections[:, 4] /= im_scalex
        detections[:, 6] /= im_scalex

        detections[:, 1] /= im_scaley
        detections[:, 3] /= im_scaley
        detections[:, 5] /= im_scaley
        detections[:, 7] /= im_scaley

      for bid, box in enumerate(detections):

        boxo = detectionso[bid]
        # score = boxo[8]
        boxr = boxo[0:8].reshape(-1, 2)
        # box_area = area(boxr.reshape(8))

        # conf_factor = score / box_area

        center = (boxr[0, :] + boxr[1, :] + boxr[2, :] + boxr[3, :]) / 4

        dw = boxr[2, :] - boxr[1, :]
        dw2 = boxr[0, :] - boxr[3, :]
        dh = boxr[1, :] - boxr[0, :]
        dh2 = boxr[3, :] - boxr[2, :]

        h = math.sqrt(dh[0] * dh[0] + dh[1] * dh[1]) + 1
        h2 = math.sqrt(dh2[0] * dh2[0] + dh2[1] * dh2[1]) + 1
        h = (h + h2) / 2
        w = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1])
        w2 = math.sqrt(dw2[0] * dw2[0] + dw2[1] * dw2[1])
        w = (w + w2) / 2

        if ((h - 1) / im_scaley) < min_height:
          continue

        input_W = im_data.size(3)
        input_H = im_data.size(2)
        target_h = norm_height

        scale = target_h / h
        target_gw = int(w * scale + target_h / 4)
        target_gw = max(8, int(round(target_gw / 8)) * 8)
        xc = center[0]
        yc = center[1]
        w2 = w
        h2 = h

        angle = math.atan2((boxr[2][1] - boxr[1][1]), boxr[2][0] - boxr[1][0])
        angle2 = math.atan2((boxr[3][1] - boxr[0][1]), boxr[3][0] - boxr[0][0])
        angle = (angle + angle2) / 2

        # show pooled image in image layer
        scalex = (w2 + h2 / 4) / input_W
        scaley = h2 / input_H

        th11 = scalex * math.cos(angle)
        th12 = -math.sin(angle) * scaley * input_H / input_W
        th13 = (2 * xc - input_W - 1) / (input_W - 1)

        th21 = math.sin(angle) * scalex * input_W / input_H
        th22 = scaley * math.cos(angle)
        th23 = (2 * yc - input_H - 1) / (input_H - 1)

        t = np.asarray([th11, th12, th13, th21, th22, th23], dtype=np.float)
        t = torch.from_numpy(t).type(torch.FloatTensor)
        t = t.to(device)
        theta = t.view(-1, 2, 3)

        grid = F.affine_grid(theta, torch.Size((1, 3, int(target_h), int(target_gw))))
        x = F.grid_sample(im_data, grid)



        features = net.forward_features(x)
        labels_pred = net.forward_ocr(features)

        ctc_f = labels_pred.data.cpu().numpy()
        ctc_f = ctc_f.swapaxes(1, 2)

        labels = ctc_f.argmax(2)

        conf = np.mean(np.exp(ctc_f.max(2)[labels > 3]))


        det_text, conf2, dec_s, word_splits = print_seq_ext(labels[0, :], codec)
        det_text = det_text.strip()

        if conf < 0.01 and len(det_text) == 3:
          continue

        if len(det_text) > 0:
          dtxt = det_text.strip()
          if len(dtxt) >= eval_text_length:
            # print('{0} - {1}'.format(dtxt, conf_factor))
            boxw = np.copy(boxr)
            boxw[:, 1] /= im_scaley
            boxw[:, 0] /= im_scalex
            boxw = boxw.reshape(8)

            detetcions_out.append([boxw, dtxt])

      pix = img

      # if args.evaluate == 1:
      tp, tp_e2e, gt_e2e, tp_e2e_ed1, detection_to_gt, pixx = evaluate_image(pix, detetcions_out, gt_rect, gt_txts,
                                                                             eval_text_length=eval_text_length)
      tp_all += tp
      gt_all += len(gt_txts)
      tp_e2e_all += tp_e2e
      gt_e2e_all += gt_e2e
      tp_e2e_ed1_all += tp_e2e_ed1
      detecitons_all += len(detetcions_out)

      if save:
        cv2.imwrite('{0}/{1}'.format(save_dir,base_nam), pixx)

      # print("	E2E recall tp_e2e:{0:.3f} / tp:{1:.3f} / e1:{2:.3f}, precision: {3:.3f}".format(
      #   tp_e2e_all / float(max(1, gt_e2e_all)),
      #   tp_all / float(max(1, gt_e2e_all)),
      #   tp_e2e_ed1_all / float(max(1, gt_e2e_all)),
      #   tp_all / float(max(1, detecitons_all))))

    note_file.write('Model{4}---E2E recall tp_e2e:{0:.3f} / tp:{1:.3f} / e1:{2:.3f}, precision: {3:.3f} \n'.format(
      tp_e2e_all / float(max(1, gt_e2e_all)),
      tp_all / float(max(1, gt_e2e_all)),
      tp_e2e_ed1_all / float(max(1, gt_e2e_all)),
      tp_all / float(max(1, detecitons_all)),name_model))

    note_file.close()
  return (
      tp_e2e_all / float(max(1, gt_e2e_all)),
      tp_all / float(max(1, gt_e2e_all)),
      tp_e2e_ed1_all / float(max(1, gt_e2e_all)),
      tp_all / float(max(1, detecitons_all)))
    # res_file.close()

def evaluate_e2e_crnn(root, net, norm_height = 48,name_model='E2E', normalize= False ,save = False, cuda= True,save_dir = 'eval'):
  #Decription : evaluate model E2E
  net = net.eval()
  # if cuda:
  #   print('Using cuda ...')
  #   net = net.to(device)


  images = glob.glob(os.path.join(root, '*.jpg'))
  png = glob.glob(os.path.join(root, '*.png'))
  images.extend(png)
  png = glob.glob(os.path.join(root, '*.JPG'))
  images.extend(png)

  imagess = np.asarray(images)

  tp_all = 0
  gt_all = 0
  tp_e2e_all = 0
  gt_e2e_all = 0
  tp_e2e_ed1_all = 0
  detecitons_all = 0
  eval_text_length = 2
  segm_thresh = 0.5
  min_height = 8
  idx =0

  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  note_path = os.path.join(save_dir, 'note_eval.txt')
  note_file = open(note_path, 'a')


  with torch.no_grad():

    index = np.arange(0, imagess.shape[0])
    # np.random.shuffle(index)
    for i in index:
      img_name = imagess[i]
      base_nam = os.path.basename(img_name)
      #
      # if args.evaluate == 1:
      res_gt = base_nam.replace(".jpg", '.txt').replace(".png", '.txt')
      res_gt = '{0}/gt_{1}'.format(root, res_gt)
      if not os.path.exists(res_gt):
        res_gt = base_nam.replace(".jpg", '.txt').replace("_", "")
        res_gt = '{0}/gt_{1}'.format(root, res_gt)
        if not os.path.exists(res_gt):
          print('missing! {0}'.format(res_gt))
          gt_rect, gt_txts = [], []
      # continue
      gt_rect, gt_txts = load_gt(res_gt)


      # print(img_name)
      img = cv2.imread(img_name)

      im_resized,_ = resize_image(img, max_size=1848 * 1024, scale_up=False)  # 1348*1024 #1848*1024
      images = np.asarray([im_resized], dtype=np.float)

      if normalize:
        images /= 128
        images -= 1
      im_data = net_utils.np_to_variable(images, is_cuda=cuda).permute(0, 3, 1, 2)

      [iou_pred, iou_pred1], rboxs, angle_pred, features = net(im_data)
      iou = iou_pred.data.cpu()[0].numpy()
      iou = iou.squeeze(0)

      rbox = rboxs[0].data.cpu()[0].numpy()
      rbox = rbox.swapaxes(0, 1)
      rbox = rbox.swapaxes(1, 2)

      detections = get_boxes(iou, rbox, angle_pred[0].data.cpu()[0].numpy(), segm_thresh)

      im_scalex = im_resized.shape[1] / img.shape[1]
      im_scaley = im_resized.shape[0] / img.shape[0]

      detetcions_out = []
      detectionso = np.copy(detections)
      if len(detections) > 0:
        detections[:, 0] /= im_scalex
        detections[:, 2] /= im_scalex
        detections[:, 4] /= im_scalex
        detections[:, 6] /= im_scalex

        detections[:, 1] /= im_scaley
        detections[:, 3] /= im_scaley
        detections[:, 5] /= im_scaley
        detections[:, 7] /= im_scaley

      for bid, box in enumerate(detections):

        boxo = detectionso[bid]
        # score = boxo[8]
        boxr = boxo[0:8].reshape(-1, 2)
        # box_area = area(boxr.reshape(8))

        # conf_factor = score / box_area

        center = (boxr[0, :] + boxr[1, :] + boxr[2, :] + boxr[3, :]) / 4

        dw = boxr[2, :] - boxr[1, :]
        dw2 = boxr[0, :] - boxr[3, :]
        dh = boxr[1, :] - boxr[0, :]
        dh2 = boxr[3, :] - boxr[2, :]

        h = math.sqrt(dh[0] * dh[0] + dh[1] * dh[1]) + 1
        h2 = math.sqrt(dh2[0] * dh2[0] + dh2[1] * dh2[1]) + 1
        h = (h + h2) / 2
        w = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1])
        w2 = math.sqrt(dw2[0] * dw2[0] + dw2[1] * dw2[1])
        w = (w + w2) / 2

        if ((h - 1) / im_scaley) < min_height:
          continue

        input_W = im_data.size(3)
        input_H = im_data.size(2)
        target_h = norm_height

        scale = target_h / h
        target_gw = int(w * scale + target_h / 4)
        target_gw = max(8, int(round(target_gw / 8)) * 8)
        xc = center[0]
        yc = center[1]
        w2 = w
        h2 = h

        angle = math.atan2((boxr[2][1] - boxr[1][1]), boxr[2][0] - boxr[1][0])
        angle2 = math.atan2((boxr[3][1] - boxr[0][1]), boxr[3][0] - boxr[0][0])
        angle = (angle + angle2) / 2

        # show pooled image in image layer
        scalex = (w2 + h2 / 4) / input_W
        scaley = h2 / input_H

        th11 = scalex * math.cos(angle)
        th12 = -math.sin(angle) * scaley * input_H / input_W
        th13 = (2 * xc - input_W - 1) / (input_W - 1)

        th21 = math.sin(angle) * scalex * input_W / input_H
        th22 = scaley * math.cos(angle)
        th23 = (2 * yc - input_H - 1) / (input_H - 1)

        t = np.asarray([th11, th12, th13, th21, th22, th23], dtype=np.float)
        t = torch.from_numpy(t).type(torch.FloatTensor)
        t = t.to(device)
        theta = t.view(-1, 2, 3)

        grid = F.affine_grid(theta, torch.Size((1, 3, int(target_h), int(target_gw))))
        x = F.grid_sample(im_data, grid)



        # features = net.forward_features(x)
        # labels_pred = net.forward_ocr(features)
        labels_pred = net.forward_ocr(x)
        labels_pred = labels_pred.permute(1, 2, 0)

        ctc_f = labels_pred.data.cpu().numpy()
        ctc_f = ctc_f.swapaxes(1, 2)

        labels = ctc_f.argmax(2)

        conf = np.mean(np.exp(ctc_f.max(2)[labels > 3]))
        if conf < 0.02:
        	continue


        det_text, conf2, dec_s, word_splits = print_seq_ext(labels[0, :], codec)
        det_text = det_text.strip()

        if conf < 0.01 and len(det_text) == 3:
          continue

        if len(det_text) > 0:
          dtxt = det_text.strip()
          if len(dtxt) >= eval_text_length:
            # print('{0} - {1}'.format(dtxt, conf_factor))
            boxw = np.copy(boxr)
            boxw[:, 1] /= im_scaley
            boxw[:, 0] /= im_scalex
            boxw = boxw.reshape(8)

            detetcions_out.append([boxw, dtxt])

      pix = img

      # if args.evaluate == 1:
      tp, tp_e2e, gt_e2e, tp_e2e_ed1, detection_to_gt, pixx = evaluate_image(pix, detetcions_out, gt_rect, gt_txts,
                                                                             eval_text_length=eval_text_length)
      tp_all += tp
      gt_all += len(gt_txts)
      tp_e2e_all += tp_e2e
      gt_e2e_all += gt_e2e
      tp_e2e_ed1_all += tp_e2e_ed1
      detecitons_all += len(detetcions_out)
      # print(gt_all)
      if save:
        cv2.imwrite('{0}/{1}'.format(save_dir,base_nam), pixx)

      # print("	E2E recall tp_e2e:{0:.3f} / tp:{1:.3f} / e1:{2:.3f}, precision: {3:.3f}".format(
      #   tp_e2e_all / float(max(1, gt_e2e_all)),
      #   tp_all / float(max(1, gt_e2e_all)),
      #   tp_e2e_ed1_all / float(max(1, gt_e2e_all)),
      #   tp_all / float(max(1, detecitons_all))))

    note_file.write('Model{4}---E2E recall tp_e2e:{0:.3f} / tp:{1:.3f} / e1:{2:.3f}, precision: {3:.3f} \n'.format(
      tp_e2e_all / float(max(1, gt_e2e_all)),
      tp_all / float(max(1, gt_e2e_all)),
      tp_e2e_ed1_all / float(max(1, gt_e2e_all)),
      tp_all / float(max(1, detecitons_all)),name_model))

    note_file.close()
  return (
      tp_e2e_all / float(max(1, gt_e2e_all)),
      tp_all / float(max(1, gt_e2e_all)),
      tp_e2e_ed1_all / float(max(1, gt_e2e_all)),
      tp_all / float(max(1, detecitons_all)))
    # res_file.close()



