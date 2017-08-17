import numpy as np
from matplotlib.pylab import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from changePoints import *

'''
    JP Clark
    K-Nearest Neighbor for CoverType
'''

def cohensKappa(tp, tn, fp, fn):
    tot = tp+tn+fp+fn
    p0 = (tp + tn)/tot
    pYes = ((tp + fp)/(tot))*((tp+fn)/(tot))
    pNo = ((fn + tn)/(tot))*((fp + tn)/(tot))
    pE = pYes + pNo
    return ((p0 - pE)/(1-pE))


def findThreshold(classes, losses):
    fpr, tpr, thresholds = roc_curve(classes, losses, pos_label = 0)
    maxSum = []

    for i in range(len(fpr)):
        maxSum.append(math.sqrt((1-tpr[i])**2 + (fpr[i])**2))
    mini = maxSum[0]
    myNumber = 0
    for i in range(len(maxSum)):
        if maxSum[i] <mini:
            mini = maxSum[i]
            myNumber = i

    return thresholds[myNumber]

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

mat =[]
for line in open('covtype.data').readlines():
    holder = line.split(',')
    i = 0
    for i in range (len(holder)):
            holder[i] = float(holder[i])
    mat.append(holder)
    
shape = np.shape(mat)
half = int(shape[0]/2)
testing = []
training1 = []
for i in range(11340):
    if mat[i][shape[1]-1] == 1:
        training1.append(mat[i][:])

shape = np.shape(training1)

for t in range(20):
    testing.append(training1[shape[0]-(20+t)][:])
    
training = training1[0:shape[0]-20][:]

dist = len(mat) -565892
for i in range(565892):
    testing.append(mat[i+dist][:])
        
testShape= np.shape(testing)
classes = []
for i in range(testShape[0]):
    if testing[i][54] == 1:
        classes.append(1)
    else:
        classes.append(0)

oppClasses = []
for i in range(len(classes)):
    if classes[i] == 1:
        oppClasses.append(1)
    else:
        oppClasses.append(0)

f = open('CovNNClass.txt', 'w')
for i in range(len(classes)):
    f.write(str(classes[i]) + ',')
f.close()


testing = np.delete(testing, shape[1]-1,1)
testing = normalize(testing)
training = np.delete(training, shape[1]-1, 1)
training = normalize(training)
print("Fitting Data")
neigh = NearestNeighbors(n_neighbors = 5)
neigh.fit(training, training)
print("Testing ... ")
losses = []
for i in range(len(testing)):
    dist = neigh.kneighbors([testing[i][:]])
    losses.append(dist[0][0][0])
print("done Losses")
f = open('CovNNLosses.txt', 'w')
for i in range(len(losses)):
    f.write(str(losses[i]) +',')
f.close()
