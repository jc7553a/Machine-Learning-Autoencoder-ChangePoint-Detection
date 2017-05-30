import tensorflow as tf
import numpy as np
import Autoencoder
import pylab
from matplotlib.pylab import *

mat =[]

for line in open('breastCancerBenign1.txt').readlines():
    holder = line.split(',')
    i = 0
    for i in range (len(holder)):
        if holder[i] == '?':
            holder[i] = 0
        else:
            holder[i] = float(holder[i])
    mat.append(holder)

mat = np.delete(mat, 0,1)


input = np.array(mat).astype('float32')

n_hidden = 7
n_features = 10

x = tf.placeholder(tf.float32 , [None , 10], name = 'x')

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
y_true = tf.placeholder(tf.float32, [None, 10], name = 'y_true')
n_sample = np.shape(input)
init = tf.global_variables_initializer()
loss = tf.reduce_mean(tf.square(y_true - y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

'Set up For batch training if needed'
'''
batch_size = min(10, 50)
sample = np.random.randint(5, batch_size)
print(sample)
'''

'Uncomment to Look at Graph and Nodes'
#print(tf.get_default_graph().as_graph_def())

losses = []

'Load Graph'
sess = tf.Session()
sess.run(init)

for i in range(100):
    umphreysmcgee = i
    for q in range(457):
        j = i
        if j > 457:
            j = j%457
        batch_xs = [input[j][0:10]]
        batch_ys = [input[j][0:10]]
    #print(sess.run(batch_xs))
        sess.run([train], feed_dict = {x: batch_xs, y_true: batch_ys})
    #print(sess.run(c))
        if q < 50:
            losses.append(sess.run(loss, feed_dict = {x: batch_xs, y_true: batch_ys}))


plot(losses)
show()
#print(losses)

'''
print("Final Loss ", sess.run(loss))
print("\n\n\n")
'''
bobweir = np.array([input[2][0:10]])
print(bobweir)
check =tf.sigmoid(tf.matmul(bobweir, w)+b)
check2 = tf.matmul(check, wo)+bo
print("Using final Weights and Biases")
print(sess.run(check2))

#check2 = np.array(check2)
print("Loss of This")
print(sess.run(tf.reduce_mean(tf.square(bobweir - check2))))


