import numpy as np
from changePoints import *
from matplotlib.pylab import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

'''Test Values with Fixed Sliding Window Size'''
def windowCalc(losses, windowSize):
    val = calcThresh(losses[0:windowSize])
    predVals1 = []
    for i in range(len(losses)):
        if losses[i] <= val:
            predVals1.append(1)
        else:
            predVals1.append(0)
        if i > windowSize:
            window = losses[i-windowSize:i]
            val = calcThresh(window)
    return predVals1


'''Test  Values with Variable Sized Sliding Window Using Change Point Detection'''
def changePointCalc(losses, windowSize):
    cp = rChangePoint(losses)
    i = 0
    while i < len(cp):
        cp[i]= int(cp[i])
        if cp[i] < windowSize:
            del cp[i]
            i -= 1
        i+=1
    cp.append(len(classes)-1)
    print(len(cp))
    window = losses[0:windowSize]
    predVals1 = []
    val = calcThresh(window)
    for i in range(windowSize):
        if losses[i] <= val:
            predVals1.append(1)
        else:
            predVals1.append(0)
    i = windowSize
    change = 0
    while i < len(losses):
        if i == cp[change]:
            if losses[i] <= val:
                window = losses[i-windowSize:i]
                val = calcThresh(window)
            change +=1
        if losses[i] <= val:
            predVals1.append(1)
        else:
            predVals1.append(0)
        window.append(losses[i])
        val = calcThresh(window)
        i+=1
    return predVals1

'''Calculate Threshold Between Classification Values'''
def calcThresh(reconstrucErrors):
    avg = np.average(reconstrucErrors)
    stdDev = np.std(reconstrucErrors)
    return (avg+1*stdDev)

'''Calc Cohens Kappa Statistic'''
def cohensKappa(tp, tn, fp, fn):
    tot = tp+tn+fp+fn
    p0 = (tp + tn)/tot
    pYes = ((tp + fp)/(tot))*((tp+fn)/(tot))
    pNo = ((fn + tn)/(tot))*((fp + tn)/(tot))
    pE = pYes + pNo
    return ((p0 - pE)/(1-pE))

'''Finds ROC Optimal Threshold Value'''
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


def getLosses():
    losses = []
    for line in open('shuttleLosses.txt').readlines():
        holder = line.split(',')
    del(holder[len(holder)-1])
    for j in range(len(holder)):
        holder[j] = holder[j].strip('[')
        holder[j] = holder[j].strip(']')
        holder[j] = float(holder[j])
    losses = holder
    return losses

def getClasses():
    classes = []
    for line in open('shuttleClassVals.txt').readlines():
        holder = line.split(',')
    del(holder[len(holder)-1])
    for j in range(len(holder)):
        holder[j] = float(holder[j])
    classes = holder
    return classes

def computeStatistics(classes, predVals):
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
    print("AUC " + str(1-roc_auc_score(classes, losses)) + '',end = '\n')



if __name__ == '__main__':

    classes = getClasses()
    losses = getLosses()
    '''Get Predicted Classification Values'''
    predVals = changePointCalc(losses, 100)
    print("Stats Using Change Point", end = '\n')
    computeStatistics(classes, predVals)
    print('--------------------------------')
    print("Stats Using Fixed Window Size")
    predVals = windowCalc(losses, 100)
    computeStatistics(classes, predVals)
