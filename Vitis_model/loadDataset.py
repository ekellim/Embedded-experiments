# https://www.tensorflow.org/tutorials/load_data/images

import numpy as np
import pandas as pd
import os.path
import cv2
from tensorflow.keras.utils import to_categorical
from config import *


def loadImage(filePath):
    # open image as BGR
    image = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)

    # normalize
    image = image/255.0

    # Return the image
    return image

def loadRowFromDataframe(filePath, label):
    # Load the image
    image = loadImage(filePath).astype(np.float32)

    image = image.reshape(128,128,1)

    # Return image and label
    return image, label

def loadDataset(df):
    # Load the training data and labels
    X, y = zip(*[loadRowFromDataframe(f,l) for f,l in zip(df['filePath'], df['classID'])])

    # Convert the y lists to categorical data
    y = to_categorical(y, num_classes=10)

    # Convert X to an array
    X = np.array(X)

    # Return the training data and the labels
    return X, y


def createDatasets(df, testFold, valFold):    
    # Create a mask to filter dataframe
    testMask = df['fold'] == testFold
    valMask = df['fold'] == valFold
    trainMask = ~(testMask | valMask)

    # Filter the dataframe for training data
    df_training = df.loc[trainMask]

    # Create the training dataset
    x_train, y_train = loadDataset(df_training)

    # Filter the datframe for validation data
    df_val = df.loc[valMask]

    # Create the validation dataset
    x_val, y_val = loadDataset(df_val)

    # Filter the dataframe for test data
    df_test = df.loc[testMask]

    # Create the test dataset
    x_test, y_test = loadDataset(df_test)

    # Return the datasets
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def loadUrbansound8k(testFold, valFold):
    # if the validation fold is not in a correct value...
    if valFold not in range(1,11):
        # Report the mistake
        print("validaiton fold must be a value between 1 and 10, setting validation fold=2")

        # Set the validation fold to 2
        valFold=2

    # if the test fold is not in a correct value...
    if testFold not in range(1,11):
        # Report the mistake
        print("Test fold must be a value between 1 and 10, setting test fold=1")

        # Set the test fold to 1
        testFold=1

    # Load the meta data file
    spec_df = pd.read_csv(os.path.join(SPEC_FILE_PATH, SPEC_META_CSV))

    # Create and return the training and test datasets
    return createDatasets(spec_df, testFold, valFold)