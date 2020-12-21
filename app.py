# Group: Techies
# Harshiv Patel, Rutvi Tilala and Mudra Suthar
# Date: 28th October

from flask_ngrok import run_with_ngrok
from flask import send_file
from flask import Flask, render_template, url_for , request
from werkzeug.utils import secure_filename
import io
import base64

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
from flask_cors import CORS
CORS(app)

#run_with_ngrok(app)   #starts ngrok when the app is run


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
from flask import Response , request , send_from_directory

import time
import shutil, os
import numpy as np
import json
import cv2
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
    path = '.'
    cv2.imwrite(os.path.join(path , 'tmp.png'), img)
    
    return img

@app.route('/')
def upload_file():
   return render_template('file_upload.html')	

      
@app.route("/uploader",methods=['GET', 'POST'])
def home():
#python3 inference.py ../images/00004844.jpg 0.35636 0.23724
    net = GazeNet()
    net = DataParallel(net)
    net.cuda()

    pretrained_dict = torch.load('../model/trained_model.pkl')
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        fname = f.filename
        print(fname)

    path2 = '.'    
    test_image_path = os.path.join(path2 , f.filename)
    xi, yi = detect_head(test_image_path)
    x = float(xi)
    y = float(yi) 
    print(test_image_path)
    # x = 0.372
    # y = 0.22267
    # 0.372,0.22267
    heatmap, p_x, p_y = test(net, test_image_path , (x, y))

    resultimg = draw_result(test_image_path, (x, y), heatmap, (p_x, p_y))
   
    #print(resultimg)
 
    img = Image.fromarray(resultimg, 'RGB')
    #print(img)
   # path = '/content/drive/My Drive/AI_PROJECT/GazeFollowing/code/static'
    # cv2.imwrite(os.path.join(path2 , 'tmp.jpg'), img)

    print(p_x, p_y)
    outim = ['tmp.png']
    for o in outim:
        shutil.copy(o, './static')
    return render_template('index.html')
   
    #resp = Response(send_filee('/content/drive/My Drive/AI_PROJECT/GazeFollowing/code/tmp.png'))
    #resp.status_code = 404
    #return resp

    #return Response(get_encoded_img('/content/drive/My Drive/AI_PROJECT/GazeFollowing/code/tmp.png'))
   
   # with open('tmp.png', 'r') as file:
   #  return file.read()
   # img = Image.open('/content/drive/My Drive/AI_PROJECT/GazeFollowing/code/tmp.png', mode='r')
   # img_byte_arr = io.BytesIO()
   # img.save(img_byte_arr, format='PNG')
   # my_encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    
   # img = open("/content/drive/My Drive/AI_PROJECT/GazeFollowing/code/tmp.png", 'rb').read()
    #response = request.post(URL, data=img, headers=headers)


    #with open('/content/drive/My Drive/AI_PROJECT/GazeFollowing/code/tmp.png', 'r') as file:
     #   return file.read()


   # data = open( ,'rb').read()
   # r = requests.post(your_url,data=data)
    
    #return send_file("/content/drive/My Drive/AI_PROJECT/GazeFollowing/code/tmp.jpg", mimetype='image/png')
    #return Response(str(draw_result('/content/drive/My Drive/AI_PROJECT/GazeFollowing/images/00000003.jpg', (x, y), heatmap, (p_x, p_y))))


#def send_filee(filename):
#    return send_from_directory('/content/drive/My Drive/AI_PROJECT/GazeFollowing/code', 'tmp.png')



#def get_encoded_img(image_path):
#    img = Image.open(image_path, mode='r')
#   img_byte_arr = io.BytesIO()
#   img.save(img_byte_arr, format='png')
#   my_encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
#   return img_byte_arr

run_with_ngrok(app)
  
if __name__ == '__main__':
    app.run()
