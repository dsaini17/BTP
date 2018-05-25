# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# =============================================================================
# This module is for testing Saved Images
# Will extend to frames from Live Feed
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Neural Network

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Activation,Convolution2D,MaxPooling2D,BatchNormalization
from keras.utils import np_utils

# =============================================================================
# from keras.models import model_from_json
# json_file = open('model2_6_classes.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# 
# =============================================================================

model = Sequential()
model.add(Convolution2D(128, (3, 3), padding='same', activation='relu',
                        input_shape=(48,48,1)))
model.add(Convolution2D(128,(3, 3), padding='same', activation='relu'))
model.add(Convolution2D(128,(3, 3), padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128,(3, 3), padding='same', activation='relu'))
model.add(Convolution2D(128,(3, 3), padding='same', activation='relu'))
model.add(Convolution2D(128,(3, 3), padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64,(3, 3), padding='same', activation='relu'))
model.add(Convolution2D(64,(3, 3), padding='same', activation='relu'))
model.add(Convolution2D(64,(3, 3), padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

# optimizer:
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

fname = 'weights.best.loss.model2.6.classes.hdf5'
model.load_weights(fname)

print(model.summary())

#0 = Angry, 1 = Disgust, 2 = Fear, 3 = Happy, 4 = Sad, 5 = Surprise, 6 = Neutral
#0 = Angry + Disgust,1 = Neutral , 2 = Fear, 3 = Happy, 4 = Sad, 5 = Surprise 

lst = [ 'ANGRY/DISGUST' , 'NEUTRAL' , 'FEAR','HAPPY', 'SAD','SURPRISE']

def get_label(pred):
    value = pred.argmax(1)
    emotion = lst[value[0]]
    print('Emotion = '+ str(emotion))
    return emotion

# Video Feed

rgb = cv2.VideoCapture(0)
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

def __get_data__():
    _, fr = rgb.read()
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.3, 5)
    return faces, fr, gray

def start_app():
    skip_frame = 10
    data = []
    flag = False
    ix = 0
    while True:
        ix += 1
        
        faces, fr, gray_fr = __get_data__()
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            
            roi = cv2.resize(fc, (48, 48))
            roi = roi/255
            roi = roi.reshape(1,48,48,1)
            pred = model.predict(roi,verbose=1)
            output_label = get_label(pred)
            cv2.putText(fr, output_label, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('Filter', fr)
    cv2.destroyAllWindows()


if __name__ == '__main__':
#   model = FacialExpressionModel("model1.json", "chkPt1.h5")  JSON Model can be executed 
    start_app()
