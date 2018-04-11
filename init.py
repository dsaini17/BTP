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

from sklearn.model_selection import train_test_split

X_train = X[0:28709,:]
X_test = X[28709:,:]
y_train = y[0:28709,:]
y_test = y[28709:,:]

# Convolution Neural Network

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Activation,Convolution2D,MaxPooling2D
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train,classes)
y_test = np_utils.to_categorical(y_test,classes)
X_train = X_train.reshape(X_train.shape[0],48,48,1)
X_test = X_test.reshape(X_test.shape[0],48,48,1)

model = Sequential()
model.add(Convolution2D(32,(3,3),input_shape=(48,48,1),activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, 
          batch_size=32, nb_epoch=10, verbose=1)

score = model.evaluate(X_test, y_test, verbose=1)