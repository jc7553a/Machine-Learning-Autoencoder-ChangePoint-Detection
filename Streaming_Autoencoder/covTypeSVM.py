import numpy as np
from sklearn import svm
from changePoints import *
from matplotlib.pylab import *
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

'''
    JP Clark
    Trains OCSVM On CoverType Data
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
classes = []
for i in range(testShape[0]):
    if testing[i][54] == 1:
        classes.append(1)
    else:
        classes.append(0)
f = open('CovOCSVMClass.txt', 'w')
for i in range(len(classes)):
    f.write(str(classes[i]) + ',')
f.close()

oppClasses = []
for i in range(len(classes)):
    if classes[i] == 1:
        oppClasses.append(0)
    else:
        oppClasses.append(1)


testing = np.delete(testing, shape[1]-1,1)
testing = normalize(testing)
testShape = np.shape(testing)
training = np.delete(training, shape[1]-1, 1)
training = normalize(training)
twoShape = np.shape(training)
print(len(training))
print(len(testing))
print("Starting OCSVM")
clf = svm.OneClassSVM(kernel = 'sigmoid')
clf.fit(training)
print("Start Testing OCSVM")
'''
predVals = clf.predict(testingData)
print(classes[0:15])
print(predVals[0:15])
1600
565912
'''
losses = clf.decision_function(testing)


for i in range(len(losses)):
    losses[i]= losses[i] *-1

f = open('CovOCSVMLoss.txt', 'w')
for i in range(len(losses)):
    f.write(str(losses[i]) + ',')
f.close()
'''
cp = rChangePoint(losses)
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
val = mean + .5*stdDev

for i in range(len(losses)):
    if losses[i] <= val:
        predVals.append(1)
    else:
        predVals.append(0)

'''


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
#print("Cohens Kappa ", cohensKappa(pos, neg, fPos, fNeg)*100)
print("AUC ", roc_auc_score(oppClasses, losses))

'''
