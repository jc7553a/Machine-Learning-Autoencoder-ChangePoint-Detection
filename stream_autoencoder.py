import tensorflow as tf
import numpy as np
import pylab
from random import randint
#import scipy
from matplotlib.pylab import *


def normalize(train):
    shape = np.shape(train)
    train = np.array(train)
    i = 0
    while i < (shape[1]):
        mean = np.average(train[0:shape[0]][:])
        stdDev = np.std(train[0:shape[0]][:])
        y = 0
        length = shape[0]
        while y < length:
            absolute = np.absolute(train[y][i])
            if absolute > mean +2*stdDev:
                train = np.delete(train, y, 0) 
                length -= 1
                y = -1
            y += 1
        shape = np.shape(train)
        i +=1
    return train


print("Training on Class 1 ")
mat =[]
for line in open('shuttleTrain1.txt').readlines():
    holder = line.split(' ')
    i = 0
    for i in range (len(holder)):
            holder[i] = float(holder[i])
    mat.append(holder)





Fours = []
for j in range(len(mat)):
    if mat[j][9] == 1:
        Fours.append(mat[j][:])
        
print(np.shape(Fours))
Fours.sort(key=lambda row: row[0:])
Fours = normalize(Fours)
print(np.shape(Fours))
print(Fours[0:2][:])



#Ones = normalize(Ones)
Fours = np.delete(Fours, [0,9],1)
#Fours = normalize(Fours)
shuttleData = np.array(Fours).astype('float32')
shape = np.shape(shuttleData)
print(shape)

mat2 =[]
for line in open('shuttleTest1.txt').readlines():
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

#print(scipy.stats.zscore(shuttleTest6))
'''
shuttleTest = np.array(class1[0:100][:]).astype('float32')
shuttleTest2 = np.array(class2[0:100][:]).astype('float32')
shuttleTest3 = np.array(class3[0:100][:]).astype('float32')
shuttleTest4 = np.array(class4[0:100][:]).astype('float32')
shuttleTest5 = np.array(class5[0:100][:]).astype('float32')
shuttleTest6 = np.array(class6[0:3][:]).astype('float32')
shuttleTest7 = np.array(class7[0:3][:]).astype('float32')

shapeTest4 = np.shape(shuttleTest4)
shapeTest3 = np.shape(shuttleTest3)
shapeTest5 = np.shape(shuttleTest5)
shapeTest = np.shape(shuttleTest)
shapeTest2 = np.shape(shuttleTest2)
shapeTest6 = np.shape(shuttleTest6)
shapeTest7 = np.shape(shuttleTest7)
'''


#print(shapeTraining)
n_hidden = 5
n_hidden2 = 2
n_hidden3 = 5
n_features = shape[1]



x = tf.placeholder(tf.float32 , [None , n_features], name = 'x')
#learnRate = tf.placeholder(tf.float32, shape = [], name = 'LR')

'Weights and Biases to Hidden Layer'

w = tf.Variable(tf.truncated_normal([n_features ,n_hidden], stddev = .001), name = 'weights_h')
b = tf.Variable(tf.truncated_normal([n_hidden], stddev = .001), name = 'biases_h')
'''
'Weights and Biases to Hidden Layer2'
w2 = tf.Variable(tf.random_normal([n_hidden ,n_hidden2]), name = 'weights_h2')
b2 = tf.Variable(tf.random_normal([n_hidden2]), name = 'biases_h2')

'Weights and Biases to Hidden Layer 3'
w3 = tf.Variable(tf.random_normal([n_hidden2 ,n_hidden3]), name = 'weights_h3')
b3 = tf.Variable(tf.random_normal([n_hidden3]), name = 'biases_h3')
'''
'Weights and Biases to Output Layer'
wo = tf.Variable(tf.truncated_normal([n_hidden, n_features],stddev = .001), name = 'weights_o')
bo = tf.Variable(tf.random_normal([n_features]), name = 'biases_o')

'Calculations for Encoder and Decoder'
encoder = tf.nn.sigmoid(tf.matmul(x, w) + b)
#hidden1 = tf.nn.sigmoid(tf.matmul(encoder, w2)+b2)
#hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, w3)+b3)
decoder = tf.matmul(encoder, wo) +bo
y = decoder



'Objective Functions'
y_true = tf.placeholder(tf.float32, [None, n_features], name = 'y_true')
loss = tf.reduce_mean(tf.square(y_true - y))
optimizer = tf.train.GradientDescentOptimizer(.00002).minimize(loss)
train2 = tf.train.GradientDescentOptimizer(.05).minimize(loss)
#train = optimizer.minimize(loss)

'Set up For batch training if needed'

'Uncomment to Look at Graph and Nodes'
#print(tf.get_default_graph().as_graph_def())

losses = []
mini_epochs = 400
batch_size = 5
total_epochs = 1500

print("\nNumber of Epochs: ", total_epochs)
print("Batch Size: ", batch_size)
print("\n")


'Load Graph'
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print(sess.run(w))

train_losses = []
print("Start Training....")
batchTot = []
#batchRate = np.float32(.000002)
for t in range(200):
    #print("New Batch")
    batch_losses = []
    for y in range(mini_epochs):
        rando = randint(0,shape[0]-5)
        batch_xs = shuttleData[rando:rando+batch_size][0:shape[1]]
        batch_ys = shuttleData[(rando):(rando+batch_size)][0:shape[1]]
        sess.run(optimizer, feed_dict = {x: batch_xs, y_true: batch_ys})
        batch_losses.append(sess.run(loss, feed_dict = {x: batch_xs, y_true: batch_ys}))
    #if t%10 ==0:
    batchTot.append(np.average(batch_losses))
plot(batchTot)
show()

print(sess.run(w))

for q in range(200):
    losses = []
    for t in range(shape[0]-1):
        temp = randint(0,shape[0]-1) 
        batch_xs = [shuttleData[temp][0:shape[1]]]
        batch_ys = [shuttleData[temp][0:shape[1]]]
        sess.run(optimizer, feed_dict = {x: batch_xs, y_true: batch_ys})
        losses.append(sess.run(loss, feed_dict = {x: batch_xs, y_true: batch_ys}))
    print(np.average(losses))


print(sess.run(w))


test_losses = []
print("")
print("Testing....")
print("")

for u in range(300):
    bobweir = np.array([shuttleTest[u][:]])
    test_act = np.array([shuttleTest[u][:]])
    #sess.run([train], feed_dict = {x: bobweir, y_true: test_act})
    check =tf.sigmoid(tf.matmul(bobweir, w)+b)
    check2 = tf.matmul(check, wo)+bo
    test_losses.append(sess.run(tf.reduce_sum(tf.square(check2 - test_act))))
    

mean = np.average(test_losses)
print("Class 1 Average Loss ", mean)
print("")


test_losses2 = []
for l in range(shapeTest2[0]):
    bobweir = np.array([shuttleTest2[l][:]])
    test_act = np.array([shuttleTest2[l][:]])
    #sess.run([train], feed_dict = {x: bobweir, y_true: test_act})
    check =tf.sigmoid(tf.matmul(bobweir, w)+b)
    check2 = tf.matmul(check, wo)+bo
    test_losses2.append(sess.run(tf.reduce_sum(tf.square(check2 - test_act))))

mean =(np.average(test_losses2))

print("Class 2 Average Loss ", mean)
print("")

test_losses3 = []
for l in range(shapeTest3[0]):
    bobweir = np.array([shuttleTest3[l][:]])
    test_act = np.array([shuttleTest3[l][:]])
    #sess.run([train], feed_dict = {x: bobweir, y_true: test_act})
    check =tf.sigmoid(tf.matmul(bobweir, w)+b)
    check2 = tf.matmul(check, wo)+bo
    test_losses3.append(sess.run(tf.reduce_sum(tf.square(check2 - test_act))))

mean =(np.average(test_losses3))


print("Class 3 Average Loss ", mean)
print("")


test_losses4 = []
for z in range(300):
    bobweir = np.array([shuttleTest4[z][:]])
    test_act = np.array([shuttleTest4[z][:]])
    #sess.run([train], feed_dict = {x: bobweir, y_true: test_act})
    check =tf.sigmoid(tf.matmul(bobweir, w)+b)
    check2 = tf.matmul(check, wo)+bo
    test_losses4.append(sess.run(tf.reduce_sum(tf.square(check2 - test_act))))

mean=(np.average(test_losses4))
print("Class 4 Average Loss ", mean)
print("")

test_losses5 = []
for v in range(300):
    bobweir = np.array([shuttleTest5[v][:]])
    test_act = np.array([shuttleTest5[v][:]])
    #sess.run([train], feed_dict = {x: bobweir, y_true: test_act})
    check =tf.sigmoid(tf.matmul(bobweir, w)+b)
    check2 = tf.matmul(check, wo)+bo
    test_losses5.append(sess.run(tf.reduce_sum(tf.square(check2 - test_act))))

mean = (np.average(test_losses5))
print("Class 5 Average ", mean)
print("")

mean = np.average(test_losses4)
stdDev = np.std(test_losses4)
threshhold = mean + stdDev



test_losses6 = []
for r in range(shapeTest6[0]):
    bobweir = np.array([shuttleTest6[r][:]])
    test_act = np.array([shuttleTest6[r][:]])
    #sess.run([train], feed_dict = {x: bobweir, y_true: test_act})
    check =tf.sigmoid(tf.matmul(bobweir, w)+b)
    check2 = tf.matmul(check, wo)+bo
    test_losses6.append(sess.run(tf.reduce_sum(tf.square(check2 - test_act))))

mean = (np.average(test_losses6))
print("Class 6 Average ", mean)
print("")

test_losses7 = []
for s in range(shapeTest7[0]):
    bobweir = np.array([shuttleTest7[s][:]])
    test_act = np.array([shuttleTest7[s][:]])
    #sess.run([train], feed_dict = {x: bobweir, y_true: test_act})
    check =tf.sigmoid(tf.matmul(bobweir, w)+b)
    check2 = tf.matmul(check, wo)+bo
    test_losses7.append(sess.run(tf.reduce_sum(tf.square(check2 - test_act))))

mean = (np.average(test_losses7))
print("Class 7 Average ", mean)
print("")


positive = 0
falsePositive = 0
negative = 0
falseNegative = 0


for j in range(len(test_losses)):
    if test_losses[j] > threshhold:
        falsePositive += 1
    else:
        positive += 1



for i in range(len(test_losses5)):
    if test_losses5[i] > threshhold:
        negative += 1
    else:
        falseNegative += 1
for i in range(len(test_losses4)):
    if test_losses4[i] > threshhold:
        negative += 1
    else:
        falseNegative += 1
for i in range(len(test_losses3)):
    if test_losses3[i] > threshhold:
        negative += 1
    else:
        falseNegative += 1

for i in range(len(test_losses2)):
    if test_losses2[i] > threshhold:
        negative += 1
    else:
        falseNegative += 1
for i in range(len(test_losses6)):
    if test_losses6[i] > threshhold:
        negative += 1
    else:
        falseNegative += 1
for i in range(len(test_losses7)):
    if test_losses7[i] > threshhold:
        negative += 1
    else:
        falseNegative += 1

print("Positive ", positive)
print("False Positive ", falsePositive)
print("Negative ", negative)
print("False Negative ", falseNegative)

