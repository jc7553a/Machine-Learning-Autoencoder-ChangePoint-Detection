import numpy as np
import tensorflow as tf
import random
from Runner import runningIt
import Autoencoder as ae
from matplotlib.pylab import *
import pandas as pd
from sklearn.metrics import roc_auc_score


'''
    JP Clark
    Trains + Test Neural Network Autoencoder for CoverType Data set
'''
def normalize(train):
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

def normalizeZScore(train):
    train = np.array(train)
    shape = np.shape(train)
    for i in range(shape[1]):
        mean = np.average(train[:][i])
        stdDev = np.std(train[:][i])
        for j in range(shape[0]):
            train[j][i] = (train[j][i] -mean)/stdDev

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
classVals = []
for i in range(testShape[0]):
    if testing[i][54] == 1:
        classVals.append(1)
    else:
        classVals.append(0)

print(len(classVals))


f = open("covTypeClass.txt", "w")
for i in range (len(classVals)):
    f.write(str(classVals[i]) +',')
f.close()



testing = np.delete(testing, shape[1]-1,1)
testing = normalize(testing)
testShape = np.shape(testing)
training = np.delete(training, shape[1]-1, 1)
training = normalize(training)
twoShape = np.shape(training)
print("Training Shape ", twoShape)
print("Testing Shape ", testShape)
n_hidden = 500



autoencoder = runningIt(training, twoShape, n_hidden, n_hidden)

test_losses = []
print(len(testing))
print(testShape)
for i in range(testShape[0]):
    batch_xs = [testing[i][0:testShape[1]]]
    test_losses.append(autoencoder.calc_total_cost(batch_xs))

print("AUC ", roc_auc_score(classVals, test_losses))


f = open("covTypeLoss.txt", "w")
for i in range (len(test_losses)):
    f.write(str(test_losses[i]) + ',')
f.close()

#plot(test_losses)
#show()

'''
cp = rChangePoint(test_losses)
for i in range (len(cp)):
    cp[i]= int(cp[i])




predVals = []
for i in range(len(cp)):
    if i ==0:
        for j in range(cp[i]):
            predVals.append(1)
    if i != 0 and i%2 ==1:
        for j in range(cp[i] - cp[i-1]):
            predVals.append(0)
    if i != 0 and i%2 == 0:
        for j in range(cp[i] - cp[i-1]):
            predVals.append(1)

print("")
print("Confusion Matrix")
print("")
#acc = (pos+neg)/(pos+neg+fPos+fNeg)
y_pred = pd.Series(predVals, name = 'Predicted')
y_act = pd.Series(classVals, name = 'Actual')
df_confusion = pd.crosstab(y_pred, y_act, rownames = ['Predicted'], colnames = ['Actual'], margins = True)
print(df_confusion)

print("")
print("Accuracy ", acc*100)

'''
