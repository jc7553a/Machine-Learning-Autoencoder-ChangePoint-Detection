import tensorflow as tf
import numpy as np
import Autoencoder as ae
from matplotlib.pylab import *
from random import randint

'''
    JP Clark
    Trains a Neural Network Autoencoder on Data Passed
    Returns the Trainined Autoencoder
'''

def training(trainingDataPassed, shapePassed, n_hiddenPassed):
    '''Creating Autoencoder'''
    myAuto = ae.Autoencoder(shapePassed[1], n_hiddenPassed)
    mini_epochs = 400
    batch_size = 5
    batchT = []
    '''Mini-Batch Training Autoencoder'''
    for t in range(700):
        bLoss = []
        for y in range(mini_epochs):
            rando = randint(0,shapePassed[0]-6)
            batch_xs = trainingDataPassed[rando:(rando+batch_size)][:]
            bLoss.append(myAuto.partial_fit(batch_xs))
        batchT.append(np.average(bLoss))

    '''Uncomment to See Graph of Losses'''    
    #plot(batchT)
    #show()
   
            
    
    batchT = []
    '''Start Online Training One at a Time'''
    for j in range(4):
        losses = []
        for k in range(shapePassed[0]):
            rando = randint(0,shapePassed[0]-1)
            batch_xs = [trainingDataPassed[rando][:]]
            losses.append(myAuto.partial_fit(batch_xs))
        batchT.append(np.average(losses))
        
    '''Unocomment To see Graph of Losses'''
    #plot(batchT)
    #show()
        
    '''Return the Traininged Neural Net'''
    return myAuto
