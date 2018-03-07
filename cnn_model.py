#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 22:58:01 2018

@author: rishab

"""
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras.regularizers import l2 #, activity_l2
import cPickle 
import scipy.misc
import scipy
from scipy import ndimage
from keras.preprocessing import image
from keras.models import Model
from keras import backend as K
import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import keras

cwd = os.getcwd()

mypath = cwd + '/images/'
train_dir = 'Train Directory/'

class_names = (listdir(mypath + train_dir))

n_class = len(class_names)

X=[]
y=[]

c = 0

img_rows = 299
img_cols = 299

nn = []

for i in class_names:
    con = listdir(mypath + train_dir + str(i))
    for j in tqdm(con):
        image  = cv2.imread(mypath + train_dir + str(i) + '/' + j)
        im = cv2.resize( image , (img_rows , img_cols ), interpolation = cv2.INTER_CUBIC)
        X.append(im)
        y.append(c)
    c+=1
    nn.append((i,c))
    

X = np.array(X)
y = np.array(y)

y = np.eye(len(np.unique(y)))[y]

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 3)


nb_epoch = 3

model = model_generate()

filepath='Model.{epoch:02d}-{val_acc:.4f}.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(Train_x)


model = Sequential()

model.add(Convolution2D(64 , 200 , 200 , border_mode='valid', input_shape=( img_rows, img_cols, 3 )))
model.add(Convolution2D(64 , 200 , 200 ))
model.add(MaxPooling3D(pool2_size=( 64 , 100 , 100 ),strides=(3, 3)))
model.add(Convolution2D(128 , 100 , 100 ))
model.add(Convolution2D(128 , 100 , 100 ))
model.add(MaxPooling3D(pool2_size=( 128 , 56 , 56 ),strides=(3, 3)))
model.add(Convolution2D(256 , 56 , 56 ))
model.add(Convolution2D(256 , 56 , 56 ))
model.add(Convolution2D(256 , 56 , 56 ))
model.add(MaxPooling3D(pool2_size=( 256 , 28 , 28 ),strides=(3, 3)))
model.add(Convolution2D(512 , 28 , 28 ))
model.add(Convolution2D(512 , 28 , 28 ))
model.add(Convolution2D(512 , 28 , 28 ))
model.add(MaxPooling3D(pool2_size=( 512 , 14 , 14 ),strides=(3, 3)))
model.add(Convolution2D(512 , 14 , 14 ))
model.add(Convolution2D(512 , 14 , 14 ))
model.add(Convolution2D(512 , 14 , 14 ))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(12 , activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.fit_generator(datagen.flow(X_train, y_train,
                    batch_size=30),
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=nb_epoch,
                    validation_data=(X_test, y_test),
                    callbacks=[checkpointer])

















