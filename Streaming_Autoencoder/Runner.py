import tensorflow as tf
import numpy as np
import Autoencoder as ae
from matplotlib.pylab import *
from random import randint
from changePoints import baron2000
from changePoints import rChangePoint
from rpy2.robjects.packages import importr

'''
    JP Clark
    Trains a Neural Network Autoencoder on Data Passed
    Returns the Trainined Autoencoder
'''

def runningIt(shuttleDataPassed, shapePassed, n_hiddenPassed , nh2):
    train_losses = []
    myAuto = ae.Autoencoder(shapePassed[1], n_hiddenPassed)
    print("Start Training Batches....")
    mini_epochs = 400
    batch_size = 5
    batchT = []
    for t in range(700):
        bLoss = []
        for y in range(mini_epochs):
            rando = randint(0,shapePassed[0]-6)
            batch_xs = shuttleDataPassed[rando:(rando+batch_size)][:]
            bLoss.append(myAuto.partial_fit(batch_xs))
        batchT.append(np.average(bLoss))
        
    #plot(batchT)
    #show()
   
            
    print("Start Streaming")
    batchT = []
    for j in range(4):
        losses = []
        for k in range(shapePassed[0]):
            rando = randint(0,shapePassed[0]-1)
            batch_xs = [shuttleDataPassed[rando][:]]
            losses.append(myAuto.partial_fit(batch_xs))
        batchT.append(np.average(losses))
    
    #plot(batchT)
    #show()
        
    print("Ended With Autoencoder")
    return myAuto
