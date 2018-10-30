import keras
from keras.models import load_model
from keras.utils  import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split

import tensorflow as tf
import numpy as np
import pandas as pd
import flask
import json
import cv2
import sys
import os

def label_vocabulary(train_dir):
    labels_dict = {}
    i = 0
    for img in os.listdir(train_dir):
        label = img.split('_')[0]
        if label not in labels_dict:
            labels_dict[label] = i
            i=i+1
    return labels_dict

def inv_map(map):
    return {v: k for k, v in map.iteritems()}

def create_train_data(train_dir):
    X = []
    Y = []
    for img in os.listdir(train_dir):
    	word_label = img.split('_')[0]
        label = labels_dict[word_label]
        path = os.path.join(train_dir, img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(np.array(img))
        Y.append(label)
    return X,Y	

def prepare_image(image, target):
	# resize the input image and preprocess it
	image = cv2.resize(image, target)
	image = np.array(image).reshape(-1, 50, 50, 1)
	image = image.astype('float32')
	image = image / 255.
	
	return image

# Image Size
IMG_SIZE = 50
# Number of classes (Number of Persons in Database)
NB_CLASSES = 26

model = Sequential()
model.add(Conv2D(32, kernel_size=(12, 12),activation='linear',padding='same',input_shape=(IMG_SIZE,IMG_SIZE,1)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(64, (6, 6), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))           
model.add(Dropout(0.3))
model.add(Dense(NB_CLASSES, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, 
    optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

# Arguments
train_dir = 'letters'

# Creating the dictionary of labels
global labels_dict
labels_dict = label_vocabulary(train_dir)

# Creating train data
train_X,train_Y = create_train_data(train_dir)
train_X = np.array(train_X).reshape(-1, IMG_SIZE,IMG_SIZE, 1)
train_X = train_X.astype('float32')
train_X = train_X / 255.

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)

# Split the training set in validation and training data 
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, 
                                            test_size=0.2, random_state=13)

model.fit(train_X, train_label, batch_size=64, epochs=60, verbose=1,
    validation_data=(valid_X, valid_label))

image = cv2.imread('test_letter.png', cv2.IMREAD_GRAYSCALE)
image = prepare_image(image, target=(50, 50))

outputs = model.predict(image)

prediction = {}
prediction['confidence'] = float(max(outputs[0]))
df = pd.DataFrame(data=outputs)
index = df.values[0].tolist().index(prediction['confidence'])
labels_dict_inv = inv_map(labels_dict)
prediction['class'] = labels_dict_inv.get(index)

print prediction