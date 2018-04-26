# -*- coding: utf-8 -*-

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Activation,Convolution2D,MaxPooling2D
from keras.utils import np_utils

classes = 7

model = Sequential()
model.add(Convolution2D(32,(3,3),input_shape=(48,48,1),activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

fname = 'weights-CNN.hdf5'
model.load_weights(fname)

# =============================================================================
# WEBCAM
# =============================================================================

import cv2
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    cv2.imshow('video out',img)
    k = cv2.waitKey(10)& 0xff

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

# =============================================================================
#     elif k%256 == 32:
#         # SPACE pressed
#         img_name = "opencv_frame_{}.png".format(img_counter)
#         cv2.imwrite(img_name, frame)
#         print("{} written!".format(img_name))
#         img_counter += 1
# 
# =============================================================================
cam.release()

cv2.destroyAllWindows()