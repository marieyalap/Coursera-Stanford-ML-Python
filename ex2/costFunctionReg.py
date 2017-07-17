from costFunction import costFunction
from numpy import log, dot
from sigmoid import sigmoid

def costFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    # Initialize some useful values
    m = len(y)   # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta

# =============================================================

    J=( -dot(y,log(sigmoid(dot(X,theta))))-dot((1-y),log(1-sigmoid(dot(X,theta)))))/m +(Lambda/(2*m))*(sum(theta**2)-theta[0]**2)

    return J




