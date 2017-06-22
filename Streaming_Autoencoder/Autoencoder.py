import tensorflow as tf
import numpy as np

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
