import tensorflow as tf
import numpy as np
import pylab
import random
from matplotlib.pylab import *

def normalize(train):
    shape = np.shape(train)
    print(shape)
    for i in range (shape[1]):
        mean = 0
        for k in range(shape[0]):
            mean += train[k][i]
        mean = mean/float(shape[0])
        std = np.std(train[:] [i])
        for j in range(shape[0]):
            train[j][i] = round(float((train[j][i] -mean)),3)
    return train


mat =[]
for line in open('shuttleTrain.txt').readlines():
    holder = line.split(' ')
    i = 0
    for i in range (len(holder)):
            holder[i] = float(holder[i])
    mat.append(holder)

Ones = []
for j in range(len(mat)):
    if mat[j][9] == 1:
        Ones.append(mat[j][:])


Ones.sort(key=lambda row: row[0:])
print(Ones[5344][:])
Ones = np.delete(Ones, [0,9],1)


shuttleData = np.array(Ones).astype('float32')
shape = np.shape(shuttleData)
print(shape)

'''
mat2 =[]
for line in open('shuttleTest.txt').readlines():
    holder = line.split(' ')
    i = 0
    for i in range (len(holder)):
            holder[i] = float(holder[i])
    mat2.append(holder)

mat2 = np.delete(mat2, 9, 1)
#normTest = normalize(mat2)
shuttleTest = np.array(mat2).astype('float32')
shuttleTest = shuttleTest[0:1000][:]
shapeTest = np.shape(shuttleTest)
'''




#print(shapeTraining)
n_hidden = 5
n_features = shape[1]



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
optimizer = tf.train.GradientDescentOptimizer(.5)
train = optimizer.minimize(loss)

'Set up For batch training if needed'

'Uncomment to Look at Graph and Nodes'
#print(tf.get_default_graph().as_graph_def())

losses = []
mini_epochs = 100
batch_size = 25
total_epochs = 350
buffer = [[None]*shape[1]]*batch_size

print("\nNumber of Epochs: ", total_epochs)
print("Batch Size: ", batch_size)
print("\n")


'Load Graph'
sess = tf.Session()
sess.run(init)


train_losses = []
print("Start Training....")

batch_number = 250
num_batch = 0


for i in range(total_epochs):
    if  num_batch < batch_number:
        for y in range(mini_epochs):
            batch_xs = shuttleData[(0+num_batch*batch_size):(batch_size+num_batch*batch_size)][0:shape[1]]
            batch_ys = shuttleData[(0+num_batch*batch_size):(batch_size+num_batch*batch_size)][0:shape[1]]
            sess.run([train], feed_dict = {x: batch_xs, y_true: batch_ys})
        num_batch += 1
    else:
        for j in range(shape[0]- batch_number*batch_size):
            temp = j + batch_number*batch_size
            batch_xs = [shuttleData[temp][0:shape[1]]]
            batch_ys = [shuttleData[temp][0:shape[1]]]
            sess.run([train], feed_dict = {x: batch_xs, y_true: batch_ys})
    if i > 250:
        train_losses.append(sess.run([loss], feed_dict = {x: batch_xs, y_true: batch_ys}))



#print(train_losses)
print(np.average(train_losses))
crazy = []
mean = np.average(train_losses)
stdDev = np.std(train_losses)
print(stdDev)

for i in range(len(train_losses)):
    if train_losses[i] > mean + 2*stdDev:
        crazy.append(train_losses[i])
print(crazy)
u = 0
nums = []
length = len(train_losses)
while u < length:
    if train_losses[u] > mean+ 2*stdDev:
        nums.append(u)
        del train_losses[u]
        length -=1
    u +=1
mean= np.average(train_losses)
print(mean)

stdDev = std(train_losses)
print(stdDev)

threshhold = mean + 2*stdDev

good = 0
bad = 0
for j in range(len(train_losses)):
    if train_losses[j] < threshhold:
        good +=1
    else:
        bad +=1
print("Good ", good)
print("Bad ", bad)
print(nums)


print("Going into Testing")

'''
test_losses = []
for u in range(goodShape[0]):
    bobweir = np.array([goodTest[u][:]])
    test_act = np.array([goodTest[u][:]])
    check =tf.sigmoid(tf.matmul(bobweir, w)+b)
    check2 = tf.matmul(check, wo)+bo
    test_losses.append(sess.run(tf.reduce_sum(tf.square(check2 - test_act))))
    sess.run([train, loss], feed_dict = {x: bobweir, y_true: test_act})
    


stdDev = np.std(test_losses)
mean = np.average(test_losses)
threshhold = 2*stdDev + mean
print("Mean = ", mean)
print("Std Dev = ", stdDev)

test_losses2 = []
for p in range(shapeTest[0]):
    bobweir = np.array([shuttleTest[p][:]])
    test_act = np.array([shuttleTest[p][:]])
    check =tf.sigmoid(tf.matmul(bobweir, w)+b)
    check2 = tf.matmul(check, wo)+bo
    test_losses2.append(sess.run(tf.reduce_sum(tf.square(check2 - test_act))))

#plot(test_losses)
#show()
u = 0
outliers = 0
length = len(test_losses)
while u < length:
    if test_losses[u] > .5*stdDev + mean:
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
threshhold = 2*stdDev + mean



good = 0
bad = 0
badIndex = []
for n in range (len(test_losses)):
    if test_losses[n] < threshhold:
        good +=1
    else:
        bad +=1
print("Good Test ", good)
print("GoodTest Bad ", bad)
good2 = 0
bad2 = 0
for a in range (len(test_losses2)):
    if test_losses2[a] > threshhold:
        good2 +=1
    else:
        bad2 +=1

print("Bad Good = " , good2)
print("Bad Bad = ", bad2)
'''

