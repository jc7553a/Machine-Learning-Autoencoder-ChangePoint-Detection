import tensorflow as tf
import numpy as np
import pylab
import random
from matplotlib.pylab import *


''' James Clark
    American University Data Science Lab
    6/12/2017
    A simple program for an Autoencoder
    To Classify Malignant Vs Benign Cancer
    Currently It is set up for Cross Validation
    with 90% train and 10% testing
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

print("Fold 10")
''' Input Benign Cancer to train on'''
mat =[]
for line in open('breastCancerBenign1.txt').readlines():
    holder = line.split(',')
    i = 0
    for i in range (len(holder)):
            holder[i] = float(holder[i])
    mat.append(holder)

'Lets get Rid of Patient Number, and Classification'
'It is a useless feature in this circumstance'

mat = np.delete(mat, [0,10], 1)
benignData = np.array(mat).astype('float32')
benignData = normalize(benignData)
'''Input Malignant Cancer Data'''

mat2 = []
for line in open('breastCancerMalignant1.txt').readlines():
    holder = line.split(',')
    i = 0
    for i in range (len(holder)):
            holder[i] = float(holder[i])
    mat2.append(holder)
    
'''Delete Patient Number and Classification'''
mat2 = np.delete(mat2,0,1)
mat2 = np.delete(mat2, [0,10], 1)
malData = np.array(mat2).astype('float32')
malData = normalize(malData)
bShape = np.shape(benignData)
mShape = np.shape(malData)

'''Testing Examples will be 10% of Data'''
numTestsBenign =  int(bShape[0]*.1)
numTestsMal = int(mShape[0]*.1)

'''Everything else ie. 90% of data will be for training'''
temp = np.zeros((400,9))
for i in range(388):
    for j in range(bShape[1]):
        temp[i][j] = benignData[i][j]
for i in range(2):
    for j in range(bShape[1]):
        temp[i+396][j] = benignData[i+396][j]
input = np.array(temp)
#input = benignData[0:(bShape[0] -numTestsBenign)][:]
testingBenign = benignData[(numTestsBenign*10):(numTestsBenign*11)][:]
testingBenign = normalize(testingBenign)
shapeTesting = np.shape(testingBenign)
testingMal = malData[(numTestsMal*10):(numTestsMal*11)][:]
testingMal= normalize(testingMal)
shapeMal = np.shape(testingMal)
shape = np.shape(input)
input = np.array(input)
print("Number of Training Benign: ", shape[0])
print("Number of Testing Benign: ", numTestsBenign)
print("Number of Testing Malignant: ", shapeMal[0])
print("\n")

'''Number of Hidden Units and Number of Features of Training Data'''
n_hidden = 10
n_features = shape[1]


'''Setting up Autoencoder'''
x = tf.placeholder(tf.float32 , [None , n_features], name = 'x')

'Weights and Biases to Hidden Layer'

w = tf.Variable(tf.random_normal([n_features ,n_hidden]), name = 'weights_h')
b = tf.Variable(tf.random_normal([n_hidden]), name = 'biases_h')


'Weights and Biases to Output Layer'
wo = tf.Variable(tf.random_normal([n_hidden, n_features]), name = 'weights_o')
bo = tf.Variable(tf.random_normal([n_features]), name = 'biases_o')

'Calculations for Encoder and Decoder'
encoder = tf.nn.sigmoid(tf.matmul(x, w) + b)
decoder = tf.matmul(encoder, wo) +bo
y = decoder



'Objective Functions'
y_true = tf.placeholder(tf.float32, [None, n_features], name = 'y_true')
init = tf.global_variables_initializer()
loss = tf.reduce_mean(tf.square(y_true - y))
optimizer = tf.train.GradientDescentOptimizer(.1)
train = optimizer.minimize(loss)

'Set up For batch training if needed'

'Uncomment to Look at Graph and Nodes'
#print(tf.get_default_graph().as_graph_def())

losses = []
epochs = 300
print("\nNumber of Epochs: ", epochs)
print("\n")

'Load Graph'
sess = tf.Session()
sess.run(init)
#print(sess.run(w))


for i in range(epochs):
    for q in range(shape[0]):
        j = randint(0,shape[0])
        batch_xs = [input[j][0:shape[1]]]
        batch_ys = [input[j][0:shape[1]]]
        sess.run([train, loss], feed_dict = {x: batch_xs, y_true: batch_ys})
        #losses.append(sess.run(loss, feed_dict = {x: batch_xs, y_true: batch_ys}))



test_losses = []
for u in range(shapeTesting[0]):
    bobweir = np.array([testingBenign[u][:]])
    test_act = np.array([testingBenign[u][:]])
    check =tf.sigmoid(tf.matmul(bobweir, w)+b)
    check2 = tf.matmul(check, wo)+bo
    test_losses.append(sess.run(tf.reduce_sum(tf.square(check2 - test_act))))

#plot(test_losses)
#show()



test_losses2 = []
for m in range(shapeMal[0]):
    bobweir2 = np.array([testingMal[m][:]])
    test_act2 = np.array([testingMal[m][:]])
    check7 =tf.sigmoid(tf.matmul(bobweir2, w)+b)
    check3 = tf.matmul(check7, wo)+bo
    test_losses2.append(sess.run(tf.reduce_mean(tf.square(check3 - test_act2))))
#print(test_losses2)
#plot(test_losses2)
#show()


goodBenign = 0
badBenign = 0
goodMal = 0
badMal = 0

print("Statistics on Original Loss")
print("Average Benign Loss ", np.average(test_losses))
print("Average Mal Loss ", np.average(test_losses2))
print("Std Dev Benign Loss ", np.std(test_losses))
print("Std Dev BenignLoss ", np.std(test_losses2))
print("\n")
stdDev = np.std(test_losses)
mean = np.average(test_losses)
threshhold = 2*stdDev + mean
u = 0
outliers = 0
length = len(test_losses)
while u < length:
    if test_losses[u] > 3*stdDev + mean:
        del test_losses[u]
        length -=1
        outliers += 1
    u +=1

print("Number of Outliers ", outliers)
stdDev = np.std(test_losses)
print("\n")
print("After Removing Outliers")
print("Std Benign Loss ", stdDev)
print("\n")
mean = np.average(test_losses)
threshhold = stdDev + mean
print("Threshold Determination: ", threshhold)
print("\n")

for b in range(len(test_losses)):
    if test_losses[b] > threshhold:
        badBenign += 1
    else:
        goodBenign += 1

for c in range(len(test_losses2)):
    if test_losses2[c] > threshhold:
        goodMal +=1
    else:
        badMal += 1

print("Positive ", goodBenign)
print("False Positive ", badBenign)
print("Negative ", goodMal)
print("False Negative ", badMal)


