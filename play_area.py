import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from tqdm import tqdm
import os
cwd = os.getcwd()

mypath = cwd + '/images/'
train_dir = 'Train Directory/'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

#print(onlyfiles)
#print(listdir(mypath + train_dir))

class_names = (listdir(mypath + train_dir))

n_class = len(class_names)

d1 = []
d2 = []
d3 = []
d4 = []

for i in class_names:
    con = listdir(mypath + train_dir + str(i))
    for j in tqdm(con):
        image  = cv2.imread(mypath + train_dir + str(i) + '/' + j)
        im = cv2.resize( image , (299 , 299 ), interpolation = cv2.INTER_CUBIC)
        im2 = im.flatten()
        d1.append(j)
        d2.append(i)
        pix = ' '.join(map(str,im2))
        d3.append(pix)

d1 = np.array([d1])
d2 = np.array([d2])
d3 = np.array([d3])



f = open("submission2.csv", "w")
f.write("{},{},{}\n".format("name", "class" , "pixels"))
for x in range(3370):
    f.write("{},{},{}\n".format(d1[0][x], d2[0][x] , d3[0][x]))
f.close()

