import os
import keras
from os import listdir
import numpy as np
from scipy.misc import imread, imresize
from keras.utils import to_categorical, np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import DatasettoNP

#Settings
img_size = 64
grayscale_images = True
num_class = 10
test_size = 0.3
batch_size = 128
epochs = 12

if __name__ == "__main__":
    try:
        #attepmt to load numpy arrays
        X = np.load('np_dataset/X.npy')
        Y = np.load('np_dataset/Y.npy')
    except:
        #if the arrays do not yet exist, go to the dataset and build them
        DatasettoNP.build_dataset('D:/TrainingData/SignLanguage/Dataset')
        X = np.load('np_dataset/X.npy')
        Y = np.load('np_dataset/Y.npy')
    #split data 70/30 after randomizing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, shuffle=True, random_state=42)
    #reshape inputs to have depth
    X_train = X_train.reshape(X_train.shape[0], 64, 64, 1)
    X_test = X_test.reshape(X_test.shape[0], 64, 64, 1)
    model = Sequential() #define which model is being used in this case it is a stack of layers
    #add input layer that is shaped to take our training data
    model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) #divide the input by half in all dimensions
    model.add(Dropout(0.25)) #randomly freeze a percentage of inputs
    model.add(Flatten()) #flatten model to single dimension
    model.add(Dense(128, activation='relu')) #add a densely connected layer
    model.add(Dropout(0.5)) #freeze more nodes at random
    model.add(Dense(num_class, activation='softmax')) #add output layer with the defined number of classes (10)
    #build the model to desired specifications
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    #train model on data
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs, verbose=1,
              validation_data=(X_test, Y_test))
    #evaluate the training
    score = model.evaluate(X_test, Y_test, verbose=0)
    print ('Test loss: ', score[0])