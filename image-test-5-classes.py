# -*- coding: utf-8 -*-

# =============================================================================
# This module is for testing Saved Images
# Will extend to frames from Live Feed
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Input Image Preprocessing Image Shape = (48,48,1) save it in variable X





# Neural Network

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Activation,Convolution2D,MaxPooling2D,BatchNormalization
from keras.utils import np_utils

classes = 5

model = Sequential()
model.add(Convolution2D(64, 5, data_format="channels_last", kernel_initializer="he_normal", 
                 input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Convolution2D(64, 4))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.5))

model.add(Convolution2D(32, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Convolution2D(32, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()

print(model.summary())

fname = 'weights.best.loss.hdf5'
model.load_weights(fname)

# x = np.reshape(resized,(1,48,48,1))
# y_pred = model.predict(X)

# print(y_pred)

# Video Feed

rgb = cv2.VideoCapture(0)
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

def workaround():
    path = 'test_img/1.jpg'
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.3, 5)
    return faces, gray

def __get_data__():
    _, fr = rgb.read()
    print(fr.shape)
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.3, 5)
    return faces, fr, gray

def start_app():
# =============================================================================
#     skip_frame = 10
#     data = []
#     flag = False
#     ix = 0
#     while True:
#         ix += 1
#         
# =============================================================================
        #faces, fr, gray_fr = __get_data__()
        faces, gray_fr = workaround()
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            
            roi = cv2.resize(fc, (48, 48))
            pred = model.predict(roi[np.newaxis, :, :, np.newaxis])
            print(pred)
            plt.imshow(img)
            #cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            #cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

# =============================================================================
#         if cv2.waitKey(1) == 27:
#             break
#         cv2.imshow('Filter', fr)
#     cv2.destroyAllWindows()
# 
# =============================================================================

if __name__ == '__main__':
#   model = FacialExpressionModel("model1.json", "chkPt1.h5")
    start_app()
