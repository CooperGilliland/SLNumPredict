import os
from os import listdir
import numpy as np
from scipy.misc import imread, imresize
from keras.utils import to_categorical


img_size = 64
grayscale_images = True
num_class = 10

#Function takes a string filepath
def getImg(dataPath):
    img = imread(dataPath, flatten=grayscale_images)
    img  =imresize(img, (img_size, img_size, 1 if grayscale_images else 3))
    return img
#function takes in the path to the training data
def build_dataset(datasetPath):
    #use directory file names as labels
    labels = listdir(datasetPath)
    X = []
    Y = []
    #iterate through current directory
    for i, label in enumerate(labels):
        datasPath = datasetPath+'/'+label
        #iterate through the contents of current folder
        for data in listdir(datasPath):
            #load each image to the x array
            img = getImg(datasPath+'/'+data)
            X.append(img)
            Y.append(i)
    #create dataset
    #normalize values to [0,1]
    X = 1-np.array(X).astype('float32')/255.
    Y = np.array(Y).astype('float32')
    Y = to_categorical(Y, num_class)
    if not os.path.exists('np_dataset/'):
        os.makedirs('np_dataset/')
    np.save('np_dataset/X.npy', X)
    np.save('np_dataset/Y.npy', Y)