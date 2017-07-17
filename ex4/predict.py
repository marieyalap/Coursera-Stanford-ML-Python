import numpy as np

from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    """ outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """

# Useful values
    m, _ = X.shape
    num_labels, _ = Theta2.shape
    X = np.column_stack((np.ones((m, 1)), X)) #
    pred1=sigmoid(np.dot(Theta1,X.T)).T
    u, _=pred1.shape
    pred1=np.column_stack((np.ones((u,1)),pred1))
    pred2=sigmoid(np.dot(Theta2,pred1.T)).T
    #pred2.shape
    p=np.argmax(pred2, axis=1)
    
    
    
    
    
    
    
    
    
    
    
    
    
# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned neural network. You should set p to a 
#               vector containing labels between 1 to num_labels.
#
# Hint: The max function might come in useful. In particular, the max
#       function can also return the index of the max element, for more
#       information see 'help max'. If your examples are in rows, then, you
#       can use max(A, [], 2) to obtain the max for each row.
#

# =========================================================================
    
    return p + 1        # add 1 to offset index of maximum in A row

