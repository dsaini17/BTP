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
# =============================================================================
# 
# dataset = pd.read_csv("fer2013/fer2013.csv")
# 
# y = dataset.iloc[:,0:1].values
# 
# X = np.zeros(shape=(dataset.shape[0],48,48))
# 
# for i in range(dataset.shape[0]):
#     a = dataset['pixels'][i].split(' ')
#     b = [int(x) for x in a]
#     c = np.asarray(b,dtype = 'float32')
#     d = c.reshape(48,48)
#     X[i] = d
#     
# X = X.astype('float32')
# X = X/255
# 
# classes = 7
# 
# def e_ind(x):
#     if(x==2 or x==5):
#         return 1
#     elif(x==4):
#         return 2
#     elif(x==6):
#         return 4
#     else:
#         return x
# F_S_D = True
# if(F_S_D):
#   y = [e_ind(x) for x in y]
#   classes = 5
#   
# y = np.reshape(y,(len(y),1))
# 
# index1 = 28709 # Cross-Validation SET ( Public Test )
# index2 = 32298 # Final Test SET ( Private Test )
# 
# X_train = X[0:index1,:]
# X_validate = X[index1:index2,:]
# X_test = X[index2:,:]
# y_train = y[0:index1,:]
# y_validate = y[index1:index2,:]
# y_test = y[index2:,:]
# =============================================================================




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


# =============================================================================
# lst = ['ANGRY' , 'DISGUST/FEAR/SURPRISED' , 'SAD' , 'HAPPY' , 'NEUTRAL']
# 
# def label(pred):
#     p = pred.argmax(1)
#     print(p.shape)
#     return lst[p[0]]
# =============================================================================

# Video Feed

rgb = cv2.VideoCapture(0)
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

# =============================================================================
# file_path = 'test_img/'
# ext = '.jpg'
# 
# def print_photo(i):
#     path = file_path+str(i)+ext
#     img = cv2.read(path)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
# 
# def workaround():
#     path = 'test_img/15.JPG'
#     img = cv2.imread(path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = facec.detectMultiScale(gray, 1.3, 5)
#     return faces, img , gray
# 
# =============================================================================
def __get_data__():
    _, fr = rgb.read()
    print(fr.shape)
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
            pred = model.predict(roi[np.newaxis, :, :, np.newaxis])
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

         if cv2.waitKey(1) == 27:
            break
         cv2.imshow('Filter', fr)
      cv2.destroyAllWindows()
      
if __name__ == '__main__':
#   model = FacialExpressionModel("model1.json", "chkPt1.h5")
    start_app()

# =============================================================================
# X = X.reshape(X.shape[0],48,48,1)
# pred_vals = model.predict(X,verbose=1)
# pred_data = pred_vals.argmax(1)
# true_data = y.argmax(1)
# 
# cnt = 0
# for i in range(0,X.shape[0]):
#     if(true_data[i] == pred_data[i]):
#         cnt+=1
# 
# print('ACC = '+ str(cnt/X.shape[0]))
# 
# =============================================================================
