import numpy as np
from matplotlib.pylab import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from changePoints import *

'''
    JP Clark
    Implement K Nearest Neighbor For Shuttle Data set
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


print("Training on Class 1 ")
print("Check Point Detection Method cpt.meanvar algorithm is SegNeigh")

'Reading in Data'
mat =[]
for line in open('shuttleTrain.txt').readlines():
    holder = line.split(' ')
    i = 0
    for i in range (len(holder)):
            holder[i] = float(holder[i])
    mat.append(holder)

'Gathering Class 1 and Sorting it by Time'
Fours = []
for j in range(len(mat)):
    if mat[j][9] == 1:
        Fours.append(mat[j][:])
Fours.sort(key=lambda row: row[0:])
shape = np.shape(Fours)


'Deleting Class Number and Time'
Fours = np.delete(Fours, [0,9],1)
Fours = normalize(Fours)

shuttleData = np.array(Fours).astype('float32')
shape = np.shape(shuttleData)

'Reaing in Testing Data'
mat2 =[]
for line in open('shuttleTest.txt').readlines():
    holder = line.split(' ')
    i = 0
    for i in range (len(holder)):
            holder[i] = float(holder[i])
    mat2.append(holder)

mat2.sort(key = lambda row: row[0:])
shape2 = np.shape(mat2)

mat3 = []
for i in range(shape2[0]):
    if len(mat3)< 1000 and mat2[i][9] == 1:
        mat3.append(mat2[i][:])
    else:
        mat3.append(mat2[i][:])
shape2 = np.shape(mat3)
classes = []
for i in range (shape2[0]):
    if mat3[i][9] == 1:
        classes.append(1)
    else:
        classes.append(0)

oppClasses = []
for i in range(len(classes)):
    if classes == 1:
        oppClasses.append(0)
    else:
        oppClasses.append(1)

mat2 = []
testingData = np.array(mat3).astype('float32')
testingData = np.delete(testingData, [0,9],1)
testingData = normalize(testingData)

neigh = NearestNeighbors(n_neighbors = 3)
neigh.fit(shuttleData, shuttleData)

losses = []
for i in range(len(testingData)):
    dist = neigh.kneighbors([testingData[i][:]])
    losses.append(dist[0][0][0])

cp = rChangePoint3(losses)
print("DOO ", len(cp))
for i in range(len(cp)):
    cp[i]= int(cp[i])
print(cp[0])
cp[len(cp)-1] = len(classes)
#plot(losses)
#show()
#cp[len(cp)-1] = cp[len(cp)-1] -1
predVals = []
switch = 0
val = findThreshold(classes[0:cp[0]], losses[0:cp[0]])

if np.average(losses[0:cp[0]]) <= val:
    switch = 1
else:
    switch = 0

for j in range(cp[0]):
    predVals.append(switch)

if switch ==1:
    val = findThreshold(classes[0:cp[0]], losses[0:cp[0]])

for i in range(len(cp)-1):
    if losses[cp[i]] <= val :
        switch = 1
    else:
        switch = 0
    for j in range(cp[i+1] - cp[i]):
        if losses[cp[i] + j] <= val:
            predVals.append(1)
        else:
            predVals.append(0)
    if switch ==1:
        val = findThreshold(classes[cp[i]:cp[i+1]], losses[cp[i]:cp[i+1]])


'''
val = findThreshold(classes, losses)
predVals = []
for i in range(len(losses)):
    if losses[i] <= val:
        predVals.append(1)
    else:
        predVals.append(0)
'''

'''
predVals = []
tTest = []
for i in range(len(classes)):
    if classes[i] == 1:
        tTest.append(losses[i])

mean = np.average(tTest)
stdDev = np.average(tTest)
val = mean + .25*stdDev

for i in range(len(losses)):
    if losses[i] <= val:
        predVals.append(1)
    else:
        predVals.append(0)
'''


pos = 0
fPos = 0
neg = 0
fNeg = 0
for i in range(len(predVals)):
    if classes[i] == 0 and classes[i] == predVals[i]:
        pos +=1
    if classes[i] == 1 and classes[i] == predVals[i]:
        neg +=1
    if classes[i] == 0 and classes[i] != predVals[i]:
        fNeg +=1
    if classes[i] ==1 and classes[i] != predVals[i]:
        fPos +=1



print("")
print("Confusion Matrix")
print("")

acc = (pos+neg)/(pos+neg+fPos+fNeg)
prec = pos/(pos+fPos)
recall = pos/(pos + fNeg)
fpRate = fPos/(neg +fPos)

y_pred = pd.Series(predVals, name = 'Predicted')
y_act = pd.Series(classes, name = 'Actual')
df_confusion = pd.crosstab(y_act, y_pred, rownames = ['Actual'], colnames = ['Predicted'], margins = True)
print(df_confusion)
print("")
print("Accuracy ", acc*100)
print("Precision ", prec*100)
print("Recall ", recall*100)
print("FP Rate ", fpRate*100)
print("Cohens Kappa ", cohensKappa(pos, neg, fPos, fNeg)*100)
print("AUC ", 1- roc_auc_score(classes, losses))
