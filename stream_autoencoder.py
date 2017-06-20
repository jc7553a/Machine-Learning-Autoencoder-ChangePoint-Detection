import tensorflow as tf
import numpy as np
from random import randint
from matplotlib.pylab import *
import pandas as pd

class Autoencoder(object):

    def __init__(self, n_features, n_hidden):
        self.n_features = n_features
        self.n_hidden = n_hidden
        #self.transfer = transfer_function

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_features])
        self.hidden = tf.sigmoid(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # cost
        self.cost = tf.reduce_mean(tf.square(self.reconstruction - self.x))
        self.optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        'Uncomment to Look at Graph and Nodes'
        #print(tf.get_default_graph().as_graph_def())


    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(tf.truncated_normal([self.n_features, self.n_hidden], stddev = .001), name = 'weights1')
        all_weights['b1'] = tf.Variable(tf.truncated_normal([self.n_hidden], stddev = .001), name = 'bias1')
        all_weights['w2'] = tf.Variable(tf.truncated_normal([self.n_hidden, self.n_features], stddev = .001), name = 'weights_o')
        all_weights['b2'] = tf.Variable(tf.truncated_normal([self.n_features], stddev = .001), name = 'biases_o')
        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X})

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])
    
    def getHiddenWeights(self):
        return self.sess.run(self.weights['w2'])
    
    def getHiddenBiases(self):
        return self.sess.run(self.weights['b2'])

    'Uncomment to Look at Graph and Nodes'
    #print(tf.get_default_graph().as_graph_def())

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


def changePoints(mat):
    delta = 0.3
    n = len(mat)
    b = -2
    c = 2
    r = 4
    d = (c-b)/(r-2)
    eps = .7
    gamma = 0.1

    xi = np.zeros((n, r))

    B = b*np.ones((n,1))
    C = c*np.ones((n,1))


    temp = np.zeros((n,1))

    for i in range(n):
        if mat[i] <= B[i][0]:
            xi[i][0] = 1
        else:
            xi[i][0] = 0
    j = 1
#while j < r:
    for h in range(n):
        if mat[h] > C[h][0]:
            xi[h][3] = 1
        else:
            xi[h][3] = 0
    #j +=1

    m = 2
    temp = np.zeros((n,1))
    temp2 = np.zeros((n,1))
    temp3 = np.zeros((n,1))

    while m < r:
        
        for i in range(n):
            if mat[i] > B[i][0]+(d*(m-2)):
                temp[i][0] = 1
            else:
                temp[i][0] = 0
        for j in range(n):
            if mat[j] <= B[i][0] +(d*(m-1)):
                temp2[j][0] = 1
            else:
                temp2[j][0] = 0
    #print(temp2)
        for t in range(n):
            temp3[t][0] = temp[t][0]*temp2[t][0]
        for q in range(n):
            xi[q][m-1] = temp3[q][0]
        m = m+1

    #print(xi)

    k = 1
    p = np.zeros((1,r))
    q = np.zeros((1,r))
    S = []

    while k < n-3:
        mean = 0
        'Find P'
        for i in range(r):
            for h in range(k):
                mean += xi[h][i]
            mean = mean/k
            p[0][i] = mean

        'Find q'
        mean = 0
        for j in range(r):
            beb = k
            while beb < n:
                mean += xi[beb][j]
                beb +=1
            mean = mean/(n-k)
            q[0][j] = mean
        #if k ==2:
            #print(p)
        'Find u'
        u = 0
        for i in range(r):
            if q[0][i] == 0:
                u+=1
        ep = eps/(n-k)

        z = np.zeros((r,1))
    #S = np.zeros(((n-3),1))
        for m in range(r):
            if p[0][m] > 0:
                if u == 0:
                    z[m][0] = p[0][m] *np.log(p[0][m]/q[0][m])
                elif q[0][m] == 0:
                    z[m][0] = p[0][m]*np.log(p[0][m]/ ep*u)
                else:
                    z[m][0] = p[0][m] *np.log(p[0][m]/q[0][m]/(1-ep))
            adder = 0
        for u in range(r):
            adder += z[u][0]
        S.append(adder) 
        k +=1


    margin = gamma*n
    t = int(margin)
    temp = -1
    while t < int(n-margin):
        if S[t] > temp:
            temp = S[t]
        t +=1
    print(temp)
    
    Smax = max(S[(int(margin)):(int(n-margin))])
    k = (int (margin))
    while k < (int(n-margin)):
        if S[k] > Smax -0.0000001:
            nuhat = k
        k +=1
    print("NUHAT")
    print(nuhat)
    W = []
    for i in range(n):
        W.append(0)

    for i in range(800):
        W[i+100] = max([0, W[i+99]+S[i+100]-S[i+99]])

    #plot(W)
    #show()


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
Fours.sort(key=lambda row: row[0:])
shape = np.shape(Fours)



Fours = np.delete(Fours, [0,9],1)
Fours = normalize(Fours)

shuttleData = np.array(Fours).astype('float32')
shape = np.shape(shuttleData)


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

myAuto = Autoencoder(shape[1], 5)
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
for i in range(4000):
    batch_xs = [shuttleData[i][0:shape[1]]]
    losses.append(myAuto.calc_total_cost(batch_xs))

cp = changePoints(losses)




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


