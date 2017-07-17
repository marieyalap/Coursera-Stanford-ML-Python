#from ex2.costFunctionReg import costFunctionReg
from numpy import log, dot
from sigmoid import sigmoid
from numpy import e #
import numpy as np

"""def sigmoid(z):
  
    g= 1./(1.+e**(-z))
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the sigmoid of each value of z (z can be a matrix,
#               vector or scalar).

# =============================================================
    return g"""



def lrCostFunction(theta, X, y, Lambda):
    """computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#
# Hint: The computation of the cost function and gradients can be
#       efficiently vectorized. For example, consider the computation
#
#           sigmoid(X * theta)
#
#       Each row of the resulting matrix will contain the value of the
#       prediction for that example. You can make use of this to vectorize
#       the cost function and gradient computations. 
#

    # =============================================================
    m = y.size
    
    J=( -dot(y,log(sigmoid(dot(X,theta))))-dot((1-y),log(1-sigmoid(dot(X,theta)))))/m +(Lambda/(2*m))*(sum(theta**2)-theta[0]**2)
    #theta0=np.copy(theta)
    #np.put(theta0,0,0)
    #J=dot((sigmoid(dot(X,theta))-y),X)/m+ (Lambda/m)*theta0
    #J= sum(-y*log(sigmoid(dot(X,theta.T)))-(1-y)*log(1-sigmoid(dot(X,theta.T))))/m
    #J= sum(-y*log(sigmoid(dot(X,theta.T)))-(1-y)*log(1-sigmoid(dot(X,theta.T))))/m
    return J




#len(theta)
#initial_theta.shape
#X.shape
#gradientvectorized (initial_theta, X, y, Lambda)
#m, n = X.shape
#initial_theta = np.zeros((n + 1, 1)) 
#lrCostFunction(initial_theta, X, y, Lambda)
