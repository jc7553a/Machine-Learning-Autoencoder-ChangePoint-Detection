import numpy as np
from changePoints import *
import random as ra
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib.pylab import *
import pandas as pd

'''
    JP Clark
    File to Test Change Point Method against
    Artificial Data
'''


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

def cohensKappa(tp, tn, fp, fn):
    tot = tp+tn+fp+fn
    p0 = (tp + tn)/tot
    pYes = ((tp + fp)/(tot))*((tp+fn)/(tot))
    pNo = ((fn + tn)/(tot))*((fp + tn)/(tot))
    pE = pYes + pNo
    return ((p0 - pE)/(1-pE))
losses = []
print("ROC")
classes =[]


#1stArtData
for i in range(10000):
    if i <= 2000:
        losses.append(ra.uniform(0, .01))
    if i >2000 and i <= 8000:
        losses.append(ra.uniform(losses[i-1]+.01, losses[i-1] + .0175))
    if i >8000:
        losses.append(ra.uniform(losses[8000]-.01, losses[8000]+.04))
    classes.append(1)

'''

#2ndArtData
for i in range(10000):
    if i < 3000:
        #losses.append(.01)
        losses.append(ra.uniform(0, 0.0001))
    if i >=3000 and  i <= 4500:
        #val = losses[i-1] +.001
        val = ra.uniform(losses[i-1]+.01, losses[i-1]+.0175)
        losses.append(val)
    if i >4500 and i <=6000:
        val = ra.uniform(losses[4500], losses[4500]+.01)
        losses.append(val)
    if i >6000 and i <= 8000:
        val = ra.uniform(losses[i-1]+.01, losses[i-1] + .0175)
        losses.append(val)
    if i > 8000:
        #losses.append(losses[7000])
        losses.append(ra.uniform(losses[8000], losses[8000]+.01))
    classes.append(1)
'''

'''
#3rdArtData
num = 1.00000001
for i in range(10000):
    if i <=2000:
        losses.append(ra.uniform(0,.0001))
    if i> 2000 and i <= 8000:
        val = ra.uniform(losses[i-1]*num, (losses[i-1]*num)+.005)
        num +=.0000001
        losses.append(val)
    if i >8000:
        losses.append(ra.uniform(losses[8000], losses[8000] + .01))
    classes.append(1)
print(len(losses))
'''
'''
#4thArtData
num = 1.00001
for i in range(10000):
    if i <= 2000:
        losses.append(ra.uniform(0, .01))
    if i >2000 and i <= 5000:
        losses.append(ra.uniform(losses[i-1]+.01, losses[i-1] + .0175))
    if i >5000 and i <= 6000:
        losses.append(ra.uniform(losses[i-1]*num, losses[i-1]*num+.005))
        num -=.000001
    if i >6000 and i <= 8000:
        losses.append(ra.uniform(losses[i-1]*num, losses[i-1]*num+.005))
        num +=.000001
    if i >8000:
        losses.append(ra.uniform(losses[8000]-.01, losses[8000]+.01))
    classes.append(1)
'''
for i in range(len(losses)-15):
    if i % 200 == 0:
        num = ra.randint(1,50)
        for j in range(num):
            losses[i+j] = losses[i+j]+ra.uniform(1.75,2.70)
            classes[i+j] = 0

oppClasses = []
for i in range(len(classes)):
    if classes[i] ==1:
        oppClasses.append(0)
    else:
        oppClasses.append(1)
f = open('artclasses.txt', 'w')
for i in range(len(classes)):
    f.write(str(classes[i]) +',')
f.close()

plot(losses)
show()
ones = []
for i in range(len(classes)):
    if classes[i] == 1:
        ones.append(losses[i])
#mean = np.average(ones)
#stdDev = np.average(ones)
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

if np.average(losses[0:cp[0]]) < val:
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

predVals = []
val = findThreshold(classes, losses)
#val = mean + stdDev
for i in range(len(losses)):
    if losses[i] <= val:
        predVals.append(1)
    else:
        predVals.append(0)


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


f = open('artData.txt', 'w')

for i in range(len(losses)):
    f.write(str(losses[i]) + ',')
f.close()


