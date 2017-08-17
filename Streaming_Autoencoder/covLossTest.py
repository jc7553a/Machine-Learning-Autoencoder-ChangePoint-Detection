import numpy as np
from changePoints import *

from matplotlib.pylab import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

'''
    JP Clark
    Testing File for CoverType on Different Learners
    Uses a ROC, Heuristic, Change Point methods
'''

def calcAuc(losses, classes):
    sortLoss = sorted(losses, reverse = True)
    sensitivity = []
    specificity = []
    maxSum = []
    for i in range(len(sortLoss)):
        thresh = sortLoss[i]
        testClass = []
        for t in range(len(losses)):
            if sortLoss[t] <= thresh:
                testClass.append(1)
            else:
                testClass.append(0)
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for t in range(len(classes)):
            if classes[t] == 0 and testClass[t] == 0:
                tp += 1
            if classes[t] == 1 and testClass[t] == 0:
                fp +=1
            if classes[t] == 0 and testClass[t] == 1:
                fn += 1
            if classes[t] == 1 and testClass[t] == 1:
                tn +=1

        TPR = tp/(tp+fn)
        FPR = (1-(tn/(tn+fp)))
        if TPR > .99:
            TPR = 1.0
        if FPR > .99:
            FPR = 1.0
        sensitivity.append(TPR)
        specificity.append(FPR)
        #maxSum.append(math.sqrt((1-TPR)**2 + (FPR)**2))

        
    return sensitivity, specificity, maxSum

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

#print("ShuttleData Set Autoencoder Change Point BinSeg")

losses = []
for line in open('CovOCSVMLoss.txt').readlines():
    holder = line.split(',')
del(holder[len(holder)-1])
for j in range(len(holder)):
    holder[j] = holder[j].strip('[')
    holder[j] = holder[j].strip(']')
    holder[j] = float(holder[j])
losses = holder


classes = []
for line in open('CovOCSVMClass.txt').readlines():
    holder = line.split(',')
del(holder[len(holder)-1])
for j in range(len(holder)):
    holder[j] = float(holder[j])
classes = holder

oppClasses = []
for i in range(len(classes)):
    if classes[i] ==1:
        oppClasses.append(0)
    else:
        oppClasses.append(1)
#classes = oppClasses
print(classes[0])
'''
revLosses = sorted(losses, reverse = True)
#sense, spec, sums = calcAuc(losses, oppClasses)
#print(sums[0:15])
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

#print(spec)
#print(sense)
#plot(fpr, tpr)
#show()
val = ((thresholds[myNumber]))

print(val)
'''
'''

val= max(sums)
for i in range(len(sums)):
    if sums[i] == val:
        myNumber = i
        break
val = revLosses[myNumber]
'''
#val =  0.0773419
#print(val)
#plot(spec, sense)
#show()
'''
mean = np.average(losses[0:100])
stdDev = np.std(losses[0:100])
val = mean + 1.75*stdDev
print(val)
cp = rChangePoint3(losses)
#print(vals)
'''

'''
tempLoss = []
changes = []
cp = []
half = int(len(losses)*.5)
for i in range(2):
    holder = rChangePoint3(losses[i*half: half + i*half])
    if i != 1:
        del holder[len(holder)-1]
    for t in range(len(holder)):
        cp.append(holder[t] + half*i)
#print(cp)
'''
'''
cp = rChangePoint3(losses)
for i in range(len(cp)):
    cp[i]= int(cp[i])
print(cp[0])
cp[len(cp)-1] = len(classes)
#plot(losses)
#show()
#cp[len(cp)-1] = cp[len(cp)-1] -1
predVals = []
switch = 0

if np.average(losses[0:cp[i]]) < val:
    switch = 1
else:
    switch = 0
for j in range(cp[0]):
    predVals.append(switch)

for i in range(len(cp)-1):
    if losses[cp[i]] <= val :
        switch = 1
    else:
        switch = 0
    for j in range(cp[i+1] - cp[i]):
        predVals.append(switch)
'''


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
    if losses[i] <= val:
        predVals.append(1)
    else:
        predVals.append(0)

if switch ==1:
    val = findThreshold(classes[0:cp[0]], losses[0:cp[0]])

for i in range(len(cp)-1):
    switch = 0
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
predVals = []
val = findThreshold(classes, losses)
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
stdDev = np.std(tTest)
val = mean -.40*stdDev

for i in range(len(losses)):
    if losses[i] <= val:
        predVals.append(1)
    else:
        predVals.append(0) 

'''



#plot(losses)
#show()

print(len(predVals))
print(len(classes))
pos = 0
fPos = 0
neg = 0
fNeg = 0
for i in range(len(classes)):
    if classes[i] == 1 and predVals[i] == 1:
        neg +=1
    if classes[i] == 0 and predVals[i] == 0:
        pos +=1
    if classes[i] == 1 and predVals[i] == 0:
        fNeg +=1
    if classes[i] ==0 and predVals[i] == 1:
        fPos +=1
print (pos)
print(fPos)
print(fNeg)
print(neg)

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
print("AUC ", roc_auc_score(oppClasses, losses))
