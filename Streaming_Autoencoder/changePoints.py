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

def baron2000(mat):
    delta = 0.175
    n = len(mat)
    b = -4
    c = 4
    #r = 20
    r= 4
    d = (c-b)/(r-2)
    eps = .7
    gamma = 0.1

    xi = np.zeros((n, r))

    B = b*np.ones((n,1))
    C = c*np.ones((n,1))
    adder = 0

    temp = np.zeros((n,1))

    for i in range(n):
        if mat[i] <= B[i][0]:
            xi[i][0] = 1
        else:
            xi[i][0] = 0
    for h in range(n):
        if mat[h] > C[h][0]:
            xi[h][3] = 1
        else:
            xi[h][3] = 0

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

        for t in range(n):
            temp3[t][0] = temp[t][0]*temp2[t][0]
        for q in range(n):
            xi[q][m-1] = temp3[q][0]
        m = m+1
    
    k = 2
    p = np.zeros((1,r))
    q = np.zeros((1,r))
    S = []

    while k <= n-2:
        
        'Find P'
        for i in range(r):
            mean = 0
            for h in range(k):
                mean += xi[h][i]
            mean = mean/(k)
            p[0][i] = mean

        'Find q'
        for j in range(r):
            mean = 0
            beb = k
            while beb < n:
                mean += xi[beb][j]
                beb +=1
            mean = mean/(n-k)
            q[0][j] = mean
        'Find u'
        u = 0
        for i in range(r):
            if q[0][i] == 0:
                u+=1
        ep = eps/(n-k)
            
        z = np.zeros((r,1))

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
        S.append((k)*adder)
        k +=1
    'Good Through Here'

    margin = (gamma*n)
    t = int(margin)
    temp = -1
    while t < int(n-margin):
        if S[t] > temp:
            temp = S[t]
        t +=1
    
    Smax = max(S[(int(margin+1)):(int(n-margin+1))])
    k = (int (margin))
    nuhat = 0
    while k < (int(n-margin)+1):
        if S[k] > Smax -0.0000001:
            nuhat = k
        k +=1
    W = []
    for i in range(n):
        W.append(0)

    #for i in range(800):
     #   W[i+100] = max([0, W[i+99]+S[i+100]-S[i+99]])
        

    #plot(W)
    #show()
    return nuhat

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
