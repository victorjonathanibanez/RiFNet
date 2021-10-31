#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------
# RiFNet prediction code 
# Author: Victor Ibanez, University of Zurich, Institute of Forensic Medicine
# Contact: victor.ibanez@uzh.ch
# -------------------------------------------------------------------------------------

import glob
import os
import imageio
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# On mac you need to shut this down
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#load model
loaded_model = tf.keras.models.load_model('RiFNet.h5')

#img directory
img_path='PMCT_images'
img_list = os.listdir(img_path)[1:]

# create folder for prediction
folder1 = 'PMCT_predictions'
frac = os.path.join(folder1, 'fractures')
no_frac = os.path.join(folder1, 'no_fractures')

if not os.path.isdir(folder1): 
  os.makedirs(folder1)
if not os.path.isdir(frac): 
  os.makedirs(frac)
if not os.path.isdir(no_frac): 
  os.makedirs(no_frac)

# predict on images
cnt_frac = 0
cnt_no_frac = 0

for i in img_list:
  img = imageio.imread(os.path.join(img_path,i))
  img2 = img[250:750, 150:1150]
  img3 = np.expand_dims(img2, axis=0)

  result = loaded_model.predict_classes(img3)

  if result == 0:
    cnt_no_frac += 1
    imageio.imwrite(os.path.join(no_frac, 'no_fracture_%d.jpg' % cnt_no_frac), img)
  elif result == 1:
    cnt_frac += 1
    imageio.imwrite(os.path.join(frac, 'fracture_%d.jpg' % cnt_frac), img)

print('All images predicted!')
