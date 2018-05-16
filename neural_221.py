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
from keras.utils import np_utils,print_summary,plot_model
from keras.callbacks import History

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
# score = model.evaluate(X_test, y_test, verbose=1)
y_pred = model.predict(X_validate,verbose = 1)

# =============================================================================
# Metrics
# =============================================================================

labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Confusion Matrix

from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,cohen_kappa_score

test_data = y_validate.argmax(1)
pred_data = y_pred.argmax(1)
cm = confusion_matrix(test_data,pred_data)
print(cm)

val = 0
for i in range(0,7):
    val += cm[i][i]
print('Validation Set Accuracy = ' + str(val/(index2-index1)))

max_val = cm.max()

plt.imshow(cm)
plt.imshow(cm,cmap='gray')

weight_list = []

for i in range(0,7):
    weight_list.append(0)

for x in pred_data:
    weight_list[x] += 1
    
nm = cm


for i in range(0,7):
    for j in range(0,7):
        nm[i][j] *= max_val
        nm[i][j] /= weight_list[i]
        
plt.imshow(cm,cmap = 'gray')        
plt.imshow(cm,interpolation='nearest') # Visualizing Confusion Matrix

# Heat Map

import seaborn as sn

df_cm = pd.DataFrame(nm,index = [labels[i] for i in range(0,7)],columns = [labels[i] for i in range(0,7)])
sn.set(font_scale = 1)
sn.heatmap(df_cm,annot = True)

# =============================================================================
# Metrics
# =============================================================================

def print_metric(score,metric):
    print(metric + ' Score is ' + str(score))

def print_all_metrics():
    print_metric(ps,'Precision')
    print_metric(rs,'Recall')
    print_metric(fs,'F1')
    print_metric(cks,'Cohen Kappa')
    
ps = precision_score(test_data,pred_data,average = 'micro')

rs = recall_score(test_data,pred_data,average = 'micro')

fs = f1_score(test_data,pred_data,average = 'micro')

cks = cohen_kappa_score(test_data,pred_data)

print_all_metrics()

# =============================================================================
# Model Architecture Visualization
# =============================================================================

from keras.utils import plot_model
plot_model(model, to_file='model.png')

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))