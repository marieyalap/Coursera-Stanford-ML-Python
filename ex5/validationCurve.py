import numpy as np

from trainLinearReg import trainLinearReg
from linearRegCostFunction import linearRegCostFunction

def validationCurve(X, y, Xval, yval):
    """returns the train
    and validation errors (in error_train, error_val)
    for different values of lambda. You are given the training set (X,
    y) and validation set (Xval, yval).
    """
    m, _ = X.shape
# Selected values of lambda (you should not change this)
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

# You need to return these variables correctly.
    error_train = np.zeros(lambda_vec.size)
    error_val = np.zeros(lambda_vec.size)

# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return training errors in 
#               error_train and the validation errors in error_val. The 
#               vector lambda_vec contains the different lambda parameters 
#               to use for each calculation of the errors, i.e, 
#               error_train(i), and error_val(i) should give 
#               you the errors obtained after training with 
#               lambda = lambda_vec(i)
#
# Note: You can loop over lambda_vec with the following:
#
#       for i = 1:length(lambda_vec)
#           lambda = lambda_vec(i)
#           # Compute train / val errors when training linear 
#           # regression with regularization parameter lambda
#           # You should store the result in error_train(i)
#           # and error_val(i)
#           ....
#           
#       end
#
#



# =========================================================================
    for j in range(lambda_vec.size) :
        
        
        theta=trainLinearReg(X, y, lambda_vec[j])
       
        
        error_train[j],_=linearRegCostFunction(X, y, theta, 0)
       
        error_val[j],_=linearRegCostFunction(Xval, yval, theta, 0) 
###===========================================================================
        
    #for i in range(1,m+1) :
        
        #for j in range(lambda_vec.size):
            #theta=trainLinearReg(X[0:i,:], y[0:i], lambda_vec[j], method='CG', maxiter=200)
       
        
            #error_train[j],_=linearRegCostFunction(X[0:i,:], y[0:i], theta, lambda_vec[j])
       
           # error_val[j],_=linearRegCostFunction(Xval, yval, theta, lambda_vec[j])      






    return lambda_vec, error_train, error_val