import numpy as np


def linearRegCostFunction(X, y, theta, Lambda):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
# Initialize some useful values

    m = y.size # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost and gradient of regularized linear 
#               regression for a particular choice of theta.
#
#               You should set J to the cost and grad to the gradient.
#

#Lambda=1
#X=np.column_stack((np.ones(m), X))
# =========================================================================
  
    # aure maniere : L2 norme
    J=(1./(2*m))*(sum((np.dot(X,theta)-y)**2))+(Lambda/(2.*m))*(sum(theta**2)-theta[0]**2)
    
    theta0=np.copy(theta)
    np.put(theta0,0,0)
    grad=np.dot(((np.dot(X,theta))-y),X)/(1.*m) + (Lambda/(1.*m))*theta0
    
        
    return J, grad






