# Group: Techies
# Harshiv Patel, Rutvi Tilala and Mudra Suthar
# Date: 20th October

import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from gazenet import GazeNet

import time
import os
import numpy as np
import json
import cv2
# import face_recognition
from PIL import Image, ImageOps
import random
from tqdm import tqdm
import operator
import itertools
from scipy.io import  loadmat
import logging
import imutils
from base64 import b64decode

from scipy import signal

from utils import data_transforms
from utils import get_paste_kernel, kernel_map

def detect_head(image_path):
  image = cv2.imread(image_path)
  image = imutils.resize(image, width=400)
  (h, w) = image.shape[:2]
  print(w,h)
  #cv2_imshow(image)
  print("[INFO] loading model...")
  prototxt = '../model/deploy.prototxt'
  model = '../model/res10_300x300_ssd_iter_140000.caffemodel'
  net = cv2.dnn.readNetFromCaffe(prototxt, model)
  image = imutils.resize(image, width=400)
  blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
  print("[INFO] computing object detections...")
  net.setInput(blob)
  detections = net.forward()
  list_x = []
  list_y = []
  for i in range(0, detections.shape[2]):

    # extract the confidence (i.e., probability) associated with the prediction
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence threshold
    if confidence > 0.5:
      # compute the (x, y)-coordinates of the bounding box for the object
      box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
      (startX, startY, endX, endY) = box.astype("int")
      iy = (startY+endY)/(2.0 * float(h))
      # draw the bounding box of the face along with the associated probability
      text = "{:.2f}%".format(confidence * 100)
      
      y = startY - 10 if startY - 10 > 10 else startY + 10
      ix = (startX+endX)/(2.0 * float(w))
        
      #print((startX+endX)/2, ((startY+endY)/2))
      print(ix,iy)
      cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
      cv2.putText(image, text, (startX, y),
      cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
      list_x.append(ix)
      list_y.append(iy)
      return ix, iy
def generate_data_field(eye_point):
    """eye_point is (x, y) and between 0 and 1"""
    height, width = 224, 224
    x_grid = np.array(range(width)).reshape([1, width]).repeat(height, axis=0)
    y_grid = np.array(range(height)).reshape([height, 1]).repeat(width, axis=1)
    grid = np.stack((x_grid, y_grid)).astype(np.float32)

    x, y = eye_point
    x, y = x * width, y * height

    grid -= np.array([x, y]).reshape([2, 1, 1]).astype(np.float32)
    norm = np.sqrt(np.sum(grid ** 2, axis=0)).reshape([1, height, width])
    # avoid zero norm
    norm = np.maximum(norm, 0.1)
    grid /= norm
    return grid

def preprocess_image(image_path, eye):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # crop face
    x_c, y_c = eye
    x_0 = x_c - 0.15
    y_0 = y_c - 0.15
    x_1 = x_c + 0.15
    y_1 = y_c + 0.15
    if x_0 < 0:
        x_0 = 0
    if y_0 < 0:
        y_0 = 0
    if x_1 > 1:
        x_1 = 1
    if y_1 > 1:
        y_1 = 1

    h, w = image.shape[:2]
    face_image = image[int(y_0 * h):int(y_1 * h), int(x_0 * w):int(x_1 * w), :]
    # process face_image for face net
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = Image.fromarray(face_image)
    face_image = data_transforms['test'](face_image)
    # process image for saliency net
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = data_transforms['test'](image)

    # generate gaze field
    gaze_field = generate_data_field(eye_point=eye)
    sample = {'image' : image,
              'face_image': face_image,
              'eye_position': torch.FloatTensor(eye),
              'gaze_field': torch.from_numpy(gaze_field)}

    return sample


def test(net, test_image_path, eye):
    net.eval()
    heatmaps = []

    data = preprocess_image(test_image_path, eye)

    image, face_image, gaze_field, eye_position = data['image'], data['face_image'], data['gaze_field'], data['eye_position']
    image, face_image, gaze_field, eye_position = map(lambda x: Variable(x.unsqueeze(0).cuda(), volatile=True), [image, face_image, gaze_field, eye_position])

    _, predict_heatmap = net([image, face_image, gaze_field, eye_position])

    final_output = predict_heatmap.cpu().data.numpy()

    heatmap = final_output.reshape([224 // 4, 224 // 4])

    h_index, w_index = np.unravel_index(heatmap.argmax(), heatmap.shape)
    f_point = np.array([w_index / 56., h_index / 56.])


    return heatmap, f_point[0], f_point[1] 

def draw_result(image_path, eye, heatmap, gaze_point):
    x1, y1 = eye
    x2, y2 = gaze_point
    im = cv2.imread(image_path)
    image_height, image_width = im.shape[:2]
    x1, y1 = image_width * x1, y1 * image_height
    x2, y2 = image_width * x2, y2 * image_height
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cv2.circle(im, (x1, y1), 5, [255, 255, 255], -1)
    cv2.circle(im, (x2, y2), 5, [255, 255, 255], -1)
    cv2.line(im, (x1, y1), (x2, y2), [255, 0, 0], 3)

    # heatmap visualization
    heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
    heatmap = np.stack([heatmap, heatmap, heatmap], axis=2)
    heatmap = cv2.resize(heatmap, (image_width, image_height))

    heatmap = (0.8 * heatmap.astype(np.float32) + 0.2 * im.astype(np.float32)).astype(np.uint8)
    img = np.concatenate((im, heatmap), axis=1)
    cv2.imwrite('tmp.png', img)
    
    return img
    
def main():

    net = GazeNet()
    net = DataParallel(net)
    net.cuda()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    pretrained_dict = torch.load('../model/trained_model.pkl')
    
    model_dict = net.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # print(pretrained_dict.summary())
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    
    test_image_path = sys.argv[1]
    xi, yi = detect_head(test_image_path)
    x = float(xi)
    y = float(yi)
    #x = float(sys.argv[2])
    #y = float(sys.argv[3])

    heatmap, p_x, p_y = test(net, test_image_path, (x, y))
    draw_result(test_image_path, (x, y), heatmap, (p_x, p_y))

    print(p_x, p_y)

    
if __name__ == '__main__':
    main()

