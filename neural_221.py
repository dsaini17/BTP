# -*- coding: utf-8 -*-

# =============================================================================
# Author : Devesh Saini
# =============================================================================

"""

 The data consists of 48x48 pixel grayscale images of faces. The faces have been
 automatically registered so that the face is more or less centered and 
 occupies about the same amount of space in each image. 
 The task is to categorize each face based on the emotion 
 shown in the facial expression in to one of seven categories 
 (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

 train.csv contains two columns, "emotion" and "pixels".
 The "emotion" column contains a numeric code ranging from 0 to 6,
 inclusive, for the emotion that is present in the image.
 The "pixels" column contains a string surrounded in quotes for each image.
 The contents of this string a space-separated pixel values in row major order. 
 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data Extraction and Pre-processing

dataset = pd.read_csv("fer2013/fer2013.csv")

y = dataset.iloc[:,0:1].values

X = np.zeros(shape=(dataset.shape[0],48,48))

for i in range(dataset.shape[0]):
    a = dataset['pixels'][i].split(' ')
    b = [int(x) for x in a]
    c = np.asarray(b,dtype = 'float32')
    d = c.reshape(48,48)
    X[i] = d
    
X = X.astype('float32')
X = X/255

classes = 7


index1 = 28709 # Cross-Validation SET ( Public Test )
index2 = 32298 # Final Test SET ( Private Test )

X_train = X[0:index1,:]
X_validate = X[index1:index2,:]
X_test = X[index2:,:]
y_train = y[0:index1,:]
y_validate = y[index1:index2,:]
y_test = y[index2:,:]

# Convolution Neural Network

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Activation,Convolution2D,MaxPooling2D,BatchNormalization
from keras.utils import np_utils

# Data Reshaping
y_train = np_utils.to_categorical(y_train,classes)
y_validate = np_utils.to_categorical(y_validate,classes)
y_test = np_utils.to_categorical(y_test,classes)
X_train = X_train.reshape(X_train.shape[0],48,48,1)
X_validate = X_validate.reshape(X_validate.shape[0],48,48,1)
X_test = X_test.reshape(X_test.shape[0],48,48,1)

model = Sequential()

# =============================================================================
# Section 1
# =============================================================================
model.add(Convolution2D(128,(4,4),input_shape=(48,48,1),strides = 1))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# =============================================================================
# Section 2
# =============================================================================
model.add(Convolution2D(128,(4,4),strides = 1))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))

# =============================================================================
# Section 3
# =============================================================================
model.add(Convolution2D(128,(4,4),strides = 1))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.2))

# =============================================================================
# Section 4
# =============================================================================
model.add(Convolution2D(128,(4,4),strides = 1))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))

# =============================================================================
# Section 5
# =============================================================================
model.add(Convolution2D(128,(4,4),strides = 1))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.2))

# =============================================================================
# Section 6
# =============================================================================
model.add(Convolution2D(128,(4,4),strides = 1))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))

# =============================================================================
# Section 7
# =============================================================================
model.add(Convolution2D(128,(2,2),strides = 1))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.2))


model.add(Flatten())

# =============================================================================
# Section 8
# =============================================================================
model.add(Dense(1024))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# =============================================================================
# Section 9
# =============================================================================
model.add(Dense(1024))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# =============================================================================
# Section 10
# =============================================================================
model.add(Dense(classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# =============================================================================
# model.fit(X_train, y_train, 
#          batch_size=512, nb_epoch=1, verbose=1)
# =============================================================================

fname = 'weights-colab-gpu.hdf5'
model.load_weights(fname)
score = model.evaluate(X_test, y_test, verbose=1)


