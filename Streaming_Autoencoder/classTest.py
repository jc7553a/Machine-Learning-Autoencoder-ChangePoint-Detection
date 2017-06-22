import tensorflow as tf
import numpy as np
from random import randint
from matplotlib.pylab import *
import pandas as pd
import Autoencoder as ae
from changePoints import baron2000

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


def anomalyDet(cp, listPassed, changePoint):
    shape = np.shape(listPassed)
    changeP = ae.Autoencoder(shape[1], 5)
    val = changePoint - int(shape[0]/2)
    losses = []
    rando = 0
    for i in range(200):
        for j in range(shape[0]):
            rando = randint(0, shape[0]-1)
            if rando == val:
                rando = rando -1
            batch_xs = [listPassed[rando][0:shape[1]]]
            changeP.partial_fit(batch_xs)
            losses.append(changeP.calc_total_cost(batch_xs))
    #val = [listPassed[int(shape[0]/2)][0:shape[1]]]
    print("Change Point Loss")
    print(changeP.calc_total_cost(cp))
    print("Mean Losses")
    print(np.average(losses))
    


print("Training on Class 1 ")

mat =[]
for line in open('shuttleTrain.txt').readlines():
    holder = line.split(' ')
    i = 0
    for i in range (len(holder)):
            holder[i] = float(holder[i])
    mat.append(holder)



Fours = []
for j in range(len(mat)):
    if mat[j][9] == 1:
        Fours.append(mat[j][:])
Fours.sort(key=lambda row: row[0:])
shape = np.shape(Fours)



Fours = np.delete(Fours, [0,9],1)
Fours = normalize(Fours)

shuttleData = np.array(Fours).astype('float32')
shape = np.shape(shuttleData)


mat2 =[]
for line in open('shuttleTest.txt').readlines():
    holder = line.split(' ')
    i = 0
    for i in range (len(holder)):
            holder[i] = float(holder[i])
    mat2.append(holder)

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

myAuto = ae.Autoencoder(shape[1], 5)
train_losses = []
print("Start Training Batches....")
batchTot = []
mini_epochs = 400
batch_size = 5

for t in range(200):
    batch_losses = []
    for y in range(mini_epochs):
        rando = randint(0,shape[0]-5)
        batch_xs = shuttleData[rando:rando+batch_size][0:shape[1]]
        batch_losses.append(myAuto.partial_fit(batch_xs))
    #if t%10 ==0:
    batchTot.append(np.average(batch_losses))
    #print(batchTot[t%10])
#plot(batchTot)
#show()


for q in range(6):
    #losses = []
    for t in range(shape[0]-1):
        temp = randint(0,shape[0]-1) 
        batch_xs = [shuttleData[temp][0:shape[1]]]
        #losses.append(myAuto.partial_fit(batch_xs))
    #print(np.average(losses))
#plot(losses)
#show()
losses = []
for i in range(5000):
    batch_xs = [shuttleData[i][0:shape[1]]]
    losses.append(myAuto.calc_total_cost(batch_xs))
plot(losses)
show()
cp = baron2000(losses)
print("Change Point", cp)
if cp != 3900:
    cpPass = [shuttleData[cp][0:shape[1]]]
    mark = shuttleData[cp-100:cp+100][0:shape[1]]
    anomalyDet(cpPass, mark,cp)
    print("")
    print(myAuto.calc_total_cost(cpPass))

'''
test_losses = []
print("")
print("Testing....")
print("")

for u in range(200):
    bobweir = np.array([shuttleTest[u][:]])
    test_losses.append(myAuto.calc_total_cost(bobweir))


test_losses2 = []
for l in range(shapeTest2[0]):
    bobweir = np.array([shuttleTest2[l][:]])
    test_losses2.append(myAuto.calc_total_cost(bobweir))



test_losses3 = []
for l in range(shapeTest3[0]):
    bobweir = np.array([shuttleTest3[l][:]])
    test_losses3.append(myAuto.calc_total_cost(bobweir))



test_losses4 = []
for z in range(200):
    bobweir = np.array([shuttleTest4[z][:]])
    test_losses4.append(myAuto.calc_total_cost(bobweir))


test_losses5 = []
for v in range(200):
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
threshhold = mean+1.5*stdDev

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


print("Positive ", positive)
print("False Positive ", falsePositive)
print("Negative ", negative)
print("False Negative ", falseNegative)

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

