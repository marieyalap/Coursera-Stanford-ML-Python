import numpy as np

import sys
sys.path.append("C:\Users\hp\Documents\GitHub\Coursera-Stanford-ML-Python\ex2")

from lrCostFunction import lrCostFunction
from gradientFunctionReg import gradientFunctionReg
from scipy.optimize import minimize
from sigmoid import sigmoid



#Lambda = 0.1 #

def gradientvectorized (theta, X, y, Lambda):
    m = y.size
    theta0=np.copy(theta)
    np.put(theta0,0,0)
    #gradvect= dot(X,(sigmoid(dot(X,theta))-y))+(Lambda/m)*theta0
    gradvect=np.dot((sigmoid(np.dot(X,theta.T))-y.T),X)/m + (Lambda/m)*theta0
    
    #gradvect=np.dot(X.T,sigmoid(np.dot(X,theta)))+(Lambda/m)*theta
    return gradvect

def oneVsAll(X, y, num_labels, Lambda):
    """trains multiple logistic regression classifiers and returns all
        the classifiers in a matrix all_theta, where the i-th row of all_theta
        corresponds to the classifier for label i
    """

    m, n = X.shape

#    all_theta = np.zeros((num_labels, n + 1))

    X = np.column_stack((np.ones((m, 1)), X))

    matY=np.ndarray(shape=(num_labels,m), dtype=float, order='F')
    initial_theta = np.zeros((n + 1, 1))   
   
    for c in range(num_labels) :
        matY[c] = [1 if i==(c+1) else 0 for i in y]
  

    theta=[]
    for y in matY :
         result=minimize(lrCostFunction, initial_theta, method='L-BFGS-B',
               jac=gradientvectorized, args=(X, y, Lambda),
               options={'gtol': 1e-4, 'disp': False, 'maxiter': 1000})      
         theta.append(result.x)
    
    all_theta=np.asarray(theta)
   

    return all_theta


