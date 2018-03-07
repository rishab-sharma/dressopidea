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
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.optimizers import SGD

cwd = os.getcwd()

mypath = cwd + '/images/'
test_dir = 'Test Directory/'

mypath = mypath + test_dir

#class_names = (listdir(mypath + train_dir))

#n_class = len(class_names)

files = listdir(mypath)

X = []
name = []

img_rows = 299
img_cols = 299

for i in tqdm(files):
	image = cv2.imread(mypath + i)
	im = cv2.resize( image , (299 , 299 ) , interpolation = cv2.INTER_CUBIC)
	X.append(im)
	name.append(i)
    
X = np.array(X)
name = np.array(name)

#________________________________________________________________________________________________________________________________________________#

base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(12, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)


model.load_weights("Model.03.hdf5")

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

print("Created model and loaded weights from file")
#________________________________________________________________________________________________________________________________________________#

classes = [('abstract', 1),
 ('Patterned', 2),
 ('Melange', 3),
 ('solid', 4),
 ('floral', 5),
 ('graphic', 6),
 ('polka dots', 7),
 ('Colourblock', 8),
 ('typography', 9),
 ('Checked', 10),
 ('Printed', 11),
 ('striped', 12)]

output = []
cl = []
for i in tqdm(X):
    pred = model.predict(i.reshape(1,299,299,3) , verbose = 0)
    output.append(str(pred[0]))
    op = np.argmax(pred[0])
    cl.append(classes[op][0])

df = pd.DataFrame({'File Name': name , 'Prediction': output, 'Predicted Class':cl})

df.to_csv('/Users/rishab/Desktop/fynd/inception-op.csv' , sep=',')








