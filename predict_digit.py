#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 20:14:04 2019

@author: coflo
"""

import pickle
import matplotlib.image as mltimg
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = mltimg.imread("1.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = cv2.resize(255 - gray,(28,28), interpolation = cv2.INTER_AREA)

var_x = []
var_y = []

# Calculate variance in image
for x in range(gray.shape[0]):
    var_x.append(np.array(gray[x,:]).var())
    var_y.append(np.array(gray[:,x]).var())
        
# Removing gradient
for x in range(gray.shape[0]):
    if var_x[x] < 5:
        gray[x] = 0
        
for y in range(gray.shape[1]):
    if var_y[y] < 34 :
        gray[:,y] = 0
        
# Removing contrast
for x in range(gray.shape[0]):
    for y in range(gray.shape[1]):
        if gray[x,y] <= 125:
            gray[x,y] = 0
            
plt.imshow(gray, cmap='Greys')

pkl_file = open('./digit_classifier.pickle', 'rb')
loaded_model = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('./lda.pickle', 'rb')
lda = pickle.load(pkl_file)
pkl_file.close()

test = gray.reshape(1,-1)
test = lda.transform(test)
yayy = loaded_model.predict(test)
print(yayy)
