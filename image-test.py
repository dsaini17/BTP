# -*- coding: utf-8 -*-

# =============================================================================
# This module is for testing Saved Images
# Will extend to frames from Live Feed
# =============================================================================

# https://keras.io/applications/#usage-examples-for-image-classification-models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 

# Input Image Preprocessing Image Shape = (48,48,1) save it in variable X

file_path = 'test_img/15.JPG'
img = cv2.imread(file_path)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#faces = face_cascade.detectMultiScale(gray, 1.3, 5) # Error
faces = [[0,100,700,900]]
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
print(plt.imshow(roi_gray))

dim_try = (48,48) 
# resize image
resized = cv2.resize(roi_gray, dim_try, interpolation = cv2.INTER_AREA)
print(plt.imshow(resized))

x = np.reshape(resized,(1,48,48,1))
y_pred = model.predict(x)
print(y_pred)
# =============================================================================
# scale_percent = 12
# width = int(roi_gray.shape[1] * scale_percent / 100)
# height = int(roi_gray.shape[0] * scale_percent / 100)
# dim = (width, height)
# =============================================================================    

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Neural Network

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Activation,Convolution2D,MaxPooling2D,BatchNormalization
from keras.utils import np_utils

classes = 7

model = Sequential()

# =============================================================================
# Section 1
# =============================================================================
model.add(Convolution2D(128,(4,4),input_shape=(48,48,1),strides = 2))


# =============================================================================
# Section 2
# =============================================================================
model.add(Convolution2D(128,(4,4),strides = 2))

# =============================================================================
# Section 3
# =============================================================================
model.add(Convolution2D(128,(4,4),strides = 2))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())


# =============================================================================
# Section 8
# =============================================================================
model.add(Dense(512))
model.add(Activation('relu'))

# =============================================================================
# Section 9
# =============================================================================
model.add(Dense(128))
model.add(Activation('relu'))

# =============================================================================
# Section 10
# =============================================================================
model.add(Dense(classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

fname = 'weights.hdf5'
model.load_weights(fname)

x = np.reshape(resized,(1,48,48,1))

y_pred = model.predict(x)

print(y_pred)


list1 = [1,2,3,4,5,6]
list2 = [9,8,5,2,1,4]

plt.plot(list1)
