from numpy import asfortranarray, squeeze, asarray
import numpy as np
from gradientFunction import gradientFunction
from numpy import dot
import sys
sys.path.append("C:\Users\hp\Documents\GitHub\Coursera-Stanford-ML-Python\ex2")
from sigmoid import sigmoid
from numpy import squeeze, asarray


def gradientFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    m = len(y)   # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the gradient of a particular choice of theta.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta


# =============================================================
    theta0=np.copy(theta)
    np.put(theta0,0,0)
    grad=dot((sigmoid(dot(X,theta.T))-y.T),X)/m+ (Lambda/m)*theta0
    
    return grad


    
