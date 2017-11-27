import numpy as np
from matplotlib.pylab import *
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import random as ra

'''
    JP Clark
    Various Change Point Detection Methods
'''


def rChangePoint(mat):
    zoo = importr('zoo' , lib_loc = "C:/Users/JP/Documents/R/win-library/3.4")
    cp = importr('changepoint' , lib_loc = "C:/Users/JP/Documents/R/win-library/3.4")
    test = robjects.r('''
                  change <- function(list){
                  list <- as.numeric(list)
                  print(typeof(list))
                  vals <- cpt.var(list, method = "PELT", class = FALSE)
                  #plot(vals)
                  return (vals)
                  }
                  '''
                  )
    r_f = robjects.r['change']
    
    res = r_f(mat)
    #print(type(res))
    myList = list(res)
    #print(myList)
    return myList

def rChangePoint2(mat):
    cpm = importr('cpm' , lib_loc = "C:/Users/JP/Documents/R/win-library/3.4")
    test = robjects.r('''
                  change <- function(list){
                  list <- as.numeric(list)
                  detectionTimes <- numeric()
                  changepoints <- numeric()
                  cpm <- makeChangePointModel(cpmType = "Lepage", ARL0 = 500)
                  i <- 0
                  while (i < length(list)){
                  i <- i +1
                  cpm <- processObservation(cpm, list[i])

                  if (changeDetected(cpm) == TRUE){
                  print(sprintf("Change Detected at observation %d", i))
                  detectionTimes <- c(detectionTimes, i)

                  Ds <- getStatistics(cpm)
                  tau <- which.max(Ds)

                  if(length(changepoints) >0){
                  tau <- tau +changepoints[length(changepoints)]
                  }
                  changepoints <- c(changepoints, tau)

                  cpm <- cpmReset(cpm)

                  i<- tau
                  }
                  }
                  return (changepoints)
                  }
                  '''
                  )
    r_f = robjects.r['change']
    
    res = r_f(mat)
    #print(type(res))
    myList = list(res)
    #print(myList)
    return myList

def rChangePoint3(mat):
    cpm = importr('cpm' , lib_loc = "C:/Users/JP/Documents/R/win-library/3.4")
    test = robjects.r('''
                change <- function(list){
                list <- as.numeric(list)
                cp <- processStream(list, "Mann-Whitney", ARL0 = 500)
                vals <- as.numeric(unlist(cp$changePoints))
                return(vals) }
                '''
                    )
    r_f = robjects.r['change']
    
    res = r_f(mat)
    #print(type(res))
    myList = list(res)
    #print(myList)
    return myList

def rChangePoint5(mat):
    rcpp = importr('Rcpp' , lib_loc = "C:/Users/Lab User/Documents/R/win-library/3.4")
    cpm = importr('ecp' , lib_loc = "C:/Users/Lab User/Documents/R/win-library/3.4")
    test = robjects.r('''
                  change <- function(list){
                  list <- as.numeric(list)
                  list <- as.matrix(list)
                  cp <- e.cp3o(list,K=1,  delta = 20, alpha =1, eps = 0.7, verbose = FALSE)
                  #cp <- as.numeric(unlist(cp))
                  return(cp$estimates)
                  }  '''
                      )
    r_f = robjects.r['change']
    
    res = r_f(mat)
    #print(type(res))
    myList = list(res)
    #print(myList)
    return myList

rpy2.robjects.numpy2ri.activate()
