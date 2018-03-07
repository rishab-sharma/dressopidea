from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
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

#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#print(onlyfiles)
#print(listdir(mypath + train_dir))

class_names = (listdir(mypath + train_dir))

n_class = len(class_names)

X=[]
y=[]

c = 0

for i in class_names:
    con = listdir(mypath + train_dir + str(i))
    for j in tqdm(con):
        image  = cv2.imread(mypath + train_dir + str(i) + '/' + j)
        im = cv2.resize( image , (299 , 299 ), interpolation = cv2.INTER_CUBIC)
        X.append(im)
        y.append(c)
    c+=1

X = np.array(X)
y = np.array(y)

y = np.eye(len(np.unique(y)))[y]

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 3)


nb_epoch = 3


filepath='Model.{epoch:02d}.hdf5'
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

datagen.fit(X_train)


base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(12, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.fit_generator(datagen.flow(X_train, y_train,
                    batch_size=30),
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch= 3,
                    validation_data=(X_test, y_test),
                    callbacks=[checkpointer])


for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

model.fit_generator(datagen.flow(X_train, y_train,
                    batch_size=30),
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=3,
                    validation_data=(X_test, y_test),
                    callbacks=[checkpointer])