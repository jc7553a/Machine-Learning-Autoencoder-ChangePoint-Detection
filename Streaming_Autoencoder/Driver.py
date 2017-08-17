import tensorflow as tf
import numpy as np
from random import randint
from matplotlib.pylab import *
import pandas as pd
import Autoencoder as ae
from sklearn.metrics import roc_auc_score
from Runner import runningIt

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

    
def createNewInstance(feat, hidden):
    'Spawns New Neural Network'
    return ae.Autoencoder(feat, hidden)

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
print("Training size ", len(shuttleData))
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

f = open('shuttleClassVals.txt', 'w')
for i in range(len(classes)):
    f.write(str(classes[i]) + ',')
f.close()

mat2 = []
testingData = np.array(mat3).astype('float32')
testingData = np.delete(testingData, [0,9],1)
testingData = normalize(testingData)

print("Testing Size ", np.shape(testingData))
'''
class1 = []
class2 = []
class3 = []
class4 = []
class5 = []
class6 = []
class7 = []

for w in range(len(mat2)):
    if mat2[w][9] == 1:
        class1.append(mat2[w][:])
    if mat2[w][9] == 2:
        class2.append(mat2[w][:])
    if mat2[w][9] == 3:
        class3.append(mat2[w][:])
    if mat2[w][9] == 4:
        class4.append(mat2[w][:])
    if mat2[w][9] == 5:
        class5.append(mat2[w][:])
    if mat2[w][9] == 6:
        class6.append(mat2[w][:])
    if mat2[w][9] == 7:
        class7.append(mat2[w][:])

class1.sort(key=lambda row: row[0:])
class2.sort(key=lambda row: row[0:])
class3.sort(key=lambda row: row[0:])
class4.sort(key=lambda row: row[0:])
class5.sort(key=lambda row: row[0:])
class6.sort(key=lambda row: row[0:])
class7.sort(key=lambda row: row[0:])

class1 = normalize(class1)
class2 = normalize(class2)
class3 = normalize(class3)
class4 = normalize(class4)
class5 = normalize(class5)
class6 = normalize(class6)
class7 = normalize(class7)

class1 = np.delete(class1, [0,9], 1)
class2 = np.delete(class2, [0,9],1)
class3 = np.delete(class3, [0,9], 1)
class4 = np.delete(class4, [0,9],1)
class5 = np.delete(class5, [0,9], 1)
class6 = np.delete(class6, [0,9], 1)
class7 = np.delete(class7, [0,9], 1)


shapeTest = np.shape(class1)
shapeTest2 = np.shape(class2)
shapeTest3 = np.shape(class3)
shapeTest4 = np.shape(class4)
shapeTest5 = np.shape(class5)
shapeTest6 = np.shape(class6)
shapeTest7 = np.shape(class7)




shuttleTest = np.array(class1[0:shapeTest[0]][:]).astype('float32')
shuttleTest2 = np.array(class2[0:shapeTest2[0]][:]).astype('float32')
shuttleTest3 = np.array(class3[0:shapeTest3[0]][:]).astype('float32')
shuttleTest4 = np.array(class4[0:shapeTest4[0]][:]).astype('float32')
shuttleTest5 = np.array(class5[0:shapeTest5[0]][:]).astype('float32')
shuttleTest6 = np.array(class6[0:shapeTest6[0]][:]).astype('float32')
shuttleTest7 = np.array(class7[0:shapeTest7[0]][:]).astype('float32')
'''








n_features = shape[1]
n_hidden = int(n_features/2)


autoencoder  = runningIt(shuttleData, shape, n_hidden, 2)
testingShape = np.shape(testingData)

test_losses = []
for i in range(testingShape[0]):
    batch_xs = [testingData[i][0:testingShape[1]]]
    test_losses.append(autoencoder.calc_total_cost(batch_xs))

f = open('shuttleLosses.txt', 'w')
for i in range(len(test_losses)):
    f.write(str(test_losses[i]) + ',')
f.close()


print("AUC ", roc_auc_score(classes, test_losses))
'''
f = open('shuttleLosses.txt', 'w')
for i in range(len(test_losses)):
    f.write(str(test_losses[i]) + ',')
f.close()

f = open('shuttleClassVals.txt', 'w')

for i in range(len(classes)):
    f.write(str(classes[i]) + ',')
f.close()

'''
'''
cp = rChangePoint(test_losses)
for i in range(len(cp)):
    cp[i] = int(cp[i])
print("")
print("Change Points Found")
print(cp)
print("")
#plot(test_losses)
#show()
predVals = []
for i in range(len(cp)):
    if i ==0:
        for j in range(cp[i]):
            predVals.append(0)
    if i != 0 and i%2 ==1:
        for j in range(cp[i] - cp[i-1]):
            predVals.append(1)
    if i != 0 and i%2 == 0:
        for j in range(cp[i] - cp[i-1]):
            predVals.append(0)

pos = 0
fPos = 0
neg = 0
fNeg = 0
print(len(predVals))
print(len(classes))
for i in range(len(predVals)):
    if classes[i] == 0 and classes[i] == predVals[i]:
        pos +=1
    if classes[i] == 1 and classes[i] == predVals[i]:
        neg +=1
    if classes[i] == 0 and classes[i] != predVals[i]:
        fNeg +=1
    if classes[i] ==1 and classes[i] != predVals[i]:
        fPos +=1
'''



'''
print("Positive ", pos)
print("Negative ", neg)
print("False Neg ", fNeg)
print("False Pos ", fPos)
'''

'''
print("")
print("Confusion Matrix")
print("")
acc = (pos+neg)/(pos+neg+fPos+fNeg)
y_pred = pd.Series(predVals, name = 'Predicted')
y_act = pd.Series(classes, name = 'Actual')
df_confusion = pd.crosstab(y_pred, y_act, rownames = ['Predicted'], colnames = ['Actual'], margins = True)
print(df_confusion)

print("")
print("Accuracy ", acc*100)


'''

'''
j = changePoint
newChange= 0
#dataToPass = []

while newChange < 34000:
    dataToPass = []
    while j < shape[0]:
        dataToPass.append(shuttleData[j][0:shape[1]])
        j +=1
    shapeToPass = np.shape(dataToPass)
    print("Shape To Pass", shapeToPass)
    autoGiven, changePoint = runningIt(dataToPass, numAutos, shapeToPass)
    newChange = newChange + changePoint
    aeList.append(autoGiven)
    numAutos += 1
    j = newChange
    print("New Change", newChange)
print("done")
print("Num Autos", numAutos)
print("Length AeList", len(aeList))


test_losses = []
print("")
print("Testing....")
print("")
count = 0
val = 1
myAuto = aeList[0]
initialData = []
for i in range(1000):
    initialData.append(shuttleData[i][0:shape[1]])
    
initialLosses =[]
for p in range(1000):
    initialLosses.append(myAuto.calc_total_cost([initialData[p][0:shape[1]]]))
threshhold = np.average(initialLosses)+2*np.std(initialLosses)

for u in range(shapeTest[0]):
    bobweir = np.array([shuttleTest[u][:]])
    test_losses.append(myAuto.calc_total_cost(bobweir))

print(val)
myAuto = aeList[0]
    
test_losses2 = []
for l in range(shapeTest2[0]):
    bobweir = np.array([shuttleTest2[l][:]])
    test_losses2.append(myAuto.calc_total_cost(bobweir))
    
test_losses3 = []
for l in range(shapeTest3[0]):
    bobweir = np.array([shuttleTest3[l][:]])
    test_losses3.append(myAuto.calc_total_cost(bobweir))
    
test_losses4 = []
for z in range(shapeTest4[0]):
    bobweir = np.array([shuttleTest4[z][:]])
    test_losses4.append(myAuto.calc_total_cost(bobweir))
    
test_losses5 = []
for v in range(shapeTest5[0]):
    bobweir = np.array([shuttleTest5[v][:]])
    test_losses5.append(myAuto.calc_total_cost(bobweir))
    
test_losses6 = []
for r in range(shapeTest6[0]):
    bobweir = np.array([shuttleTest6[r][:]])
    test_losses6.append(myAuto.calc_total_cost(bobweir))
    
test_losses7 = []
for s in range(shapeTest7[0]):
    bobweir = np.array([shuttleTest7[s][:]])
    test_losses7.append(myAuto.calc_total_cost(bobweir))
    
positive = 0
falsePositive = 0
negative = 0
falseNegative = 0
mean = 0
for i in range(len(test_losses)):
    mean = mean + test_losses[i]
mean = mean /len(test_losses)
stdDev = np.std(test_losses)
threshhold = mean+3.7*stdDev
conf = []
for j in range(len(test_losses)):
    if test_losses[j] > threshhold:
        falseNegative += 1
        conf.append(1)
    else:
        positive += 1
        conf.append(0)
for i in range(len(test_losses4)):
    if test_losses4[i] > threshhold:
        negative +=1
        conf.append(1)
    else:
        falsePositive += 1
        conf.append(0)
        
for i in range(len(test_losses5)):
    if test_losses5[i] > threshhold:
        negative += 1
        conf.append(1)
    else:
        falsePositive += 1
        conf.append(0)
for i in range(len(test_losses3)):
    if test_losses3[i] > threshhold:
        negative += 1
        conf.append(1)
    else:
        falsePositive += 1
        conf.append(0)
for i in range(len(test_losses2)):
    if test_losses2[i] > threshhold:
        negative += 1
        conf.append(1)
    else:
        falsePositive += 1
        conf.append(0)
for i in range(len(test_losses6)):
    if test_losses6[i] > threshhold:
        negative += 1
        conf.append(1)
    else:
        falsePositive += 1
        conf.append(0)
        
for i in range(len(test_losses7)):
    if test_losses7[i] > threshhold:
        negative += 1
        conf.append(1)
    else:
        falsePositive += 1
        conf.append(0)
acts = []
for i in range(len(test_losses)):
    acts.append(0)
longL = len(test_losses4)+len(test_losses2)+len(test_losses3)+len(test_losses5)+len(test_losses6)+len(test_losses7)
for j in range(longL):
    acts.append(1)
tot = negative+positive +falsePositive + falseNegative
pos = negative + positive
acc = float(pos/tot)
y_pred = pd.Series(conf, name = 'Predicted')
y_act = pd.Series(acts, name = 'Actual')
df_confusion = pd.crosstab(y_pred, y_act, rownames = ['Actual'], colnames = ['Predicted'], margins = True)
print(df_confusion)
print("")
print("True Positive ", positive)
print("False Positive ", falsePositive)
print("True Negative ", negative)
print("False Negative ", falseNegative)
print("")
print("Accuracy % = ", acc*100)
print("")
'''
