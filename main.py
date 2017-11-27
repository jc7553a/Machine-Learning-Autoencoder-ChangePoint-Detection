import numpy as np
from random import randint
from matplotlib.pylab import *
import Autoencoder as ae
from TrainingProgram import training

'''
    James Clark
    6/23/2017
    Autoencoder For Shuttle Data set
'''


def normalize(train):
    'Normalize Data by Y-Min/Max-min'''
    train = np.array(train)
    shape = np.shape(train)
    k = 0
    while k <shape[1]:
        maxim = 0
        minim = 0
        for i in range(shape[0]):
            if train[i][k] > maxim:
                maxim = train[i][k]
            if train[i][k] < minim:
                minim = train[i][k]
        denom = maxim - minim
        if denom == 0:
            denom = .0000000001
        for t in range(shape[0]):
            train[t][k] = (train[t][k] - minim)/denom
        k +=1
    return train

def getTrainingData():
    mat =[]
    for line in open('shuttleTrain.txt').readlines():
        holder = line.split(' ')
        for i in range (len(holder)):
            holder[i] = float(holder[i])
        mat.append(holder)
    training = []
    for j in range(len(mat)):
        if mat[j][9] == 1:
            training.append(mat[j][:])
    training.sort(key=lambda row: row[0:])
    'Deleting Class Number and Time'
    training = np.delete(training, [0,9],1)
    training = normalize(training)
    return np.array(training).astype('float32')

def getTestingData():  
    'Reading in Testing Data'
    mat =[]
    for line in open('shuttleTrain.txt').readlines():
        holder = line.split(' ')
        i = 0
        for i in range (len(holder)):
                holder[i] = float(holder[i])
        mat.append(holder)
    '''Sorting Data by Time'''    
    mat2.sort(key = lambda row: row[0:])
    shape2 = np.shape(mat2)
    '''We want to make sure we start off Change Point with all Normal Class'''
    mat3 = []
    for i in range(shape2[0]):
        if len(mat3)< 1000 and mat2[i][9] == 1:
            mat3.append(mat2[i][:])
        else:
            mat3.append(mat2[i][:])
    shape2 = np.shape(mat3)
    '''Writing Classification Values to File for Testing Later'''
    classes = []
    for i in range (shape2[0]):
        if mat3[i][9] == 1:
            classes.append(1)
        else:
            classes.append(0)
    f = open('shuttleClassVals.txt', 'w')
    for i in range(len(classes)):
        f.write(str(classes[i]) + ',')
    f.close()
    return np.array(mat3).astype('float32')

if __name__ == '__main__':
    trainingeData = getTrainingData()
    trainingShape = np.shape(trainingData)
    testingData = getTestingData()
    testingData = np.delete(testingData, [0,9],1)
    testingData = normalize(testingData)
    n_features = trainingShape[1] #Size of feature vector
    n_hidden = int(n_features/2) #Number of Hidden Units in Autoencoder

    '''Training Autoencoder with Training Data'''
    autoencoder  = runningIt(trainingData, trainingShape, n_hidden)
    testingShape = np.shape(testingData)

    test_losses = []
    for i in range(testingShape[0]):
        batch_xs = [testingData[i][0:testingShape[1]]]
        test_losses.append(autoencoder.calc_total_cost(batch_xs))
        

    f = open('shuttleLosses.txt', 'w')
    for i in range(len(test_losses)):
        f.write(str(test_losses[i]) + ',')
    f.close()
