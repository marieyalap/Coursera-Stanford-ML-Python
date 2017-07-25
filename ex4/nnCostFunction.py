import numpy as np
from numpy import log, dot
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):

    """computes the cost and gradient of the neural network. The
  parameters for the neural network are "unrolled" into the vector
  nn_params and need to be converted back into the weight matrices.

  The returned parameter grad should be a "unrolled" vector of the
  partial derivatives of the neural network.
    """

# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
# for our 2 layer neural network
# Obtain Theta1 and Theta2 back from nn_params
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                       (hidden_layer_size, input_layer_size + 1), order='F').copy()

    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                       (num_labels, (hidden_layer_size + 1)), order='F').copy()

 

# Setup some useful variables
    m, _ = X.shape
   

# ====================== YOUR CODE HERE ======================
# Instructions: You should complete the code by working through the
#               following parts.
#
# Part 1: Feedforward the neural network and return the cost in the
#         variable J. After implementing Part 1, you can verify that your
#         cost function computation is correct by verifying the cost
#         computed in ex4.m
#
# Part 2: Implement the backpropagation algorithm to compute the gradients
#         Theta1_grad and Theta2_grad. You should return the partial derivatives of
#         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
#         Theta2_grad, respectively. After implementing Part 2, you can check
#         that your implementation is correct by running checkNNGradients
#
#         Note: The vector y passed into the function is a vector of labels
#               containing values from 1..K. You need to map this vector into a 
#               binary vector of 1's and 0's to be used with the neural network
#               cost function.
#
#         Hint: We recommend implementing backpropagation using a for-loop
#               over the training examples if you are implementing it for the 
#               first time.
#
# Part 3: Implement regularization with the cost function and gradients.
#
#         Hint: You can implement this around the code for
#               backpropagation. That is, you can compute the gradients for
#               the regularization separately and then add them to Theta1_grad
#               and Theta2_grad from Part 2.
#



    # -------------------------------------------------------------

    # =========================================================================

##Partie 1 :
   
    MatY=np.ndarray(shape=(num_labels,m), dtype=float, order='F')
    for col in range(num_labels) :
        MatY[col] = [1 if i==(col+1) else 0 for i in y]
        
    MatY
    MatY.shape
    # feetforward algorithm
    
    X1 = np.column_stack((np.ones((m, 1)), X)) # a(1)
    pred1=sigmoid(np.dot(Theta1,X1.T)).T    ##a(2)
    u, _=pred1.shape
    pred1=np.column_stack((np.ones((u,1)),pred1))
    pred2=sigmoid(np.dot(Theta2,pred1.T)).T   ##a(3)
    
    Theta1b=Theta1[:,1:]
    Theta2b=Theta2[:,1:]
    w=np.dot(Theta1b.T,Theta1b)
    c=np.dot(Theta2b.T,Theta2b)

    a=-np.dot(MatY,log(pred2))
    b=np.dot((1-MatY),log(1-pred2))
    Theta2.shape    
        
    J=(sum(np.diagonal(a)) - sum(np.diagonal(b)))/m + (Lambda/(2*m))*(sum(np.diagonal(c))+sum(np.diagonal(w)))

    # Backpropagation algorithm

    delta2=0 
    delta1 = 0
    
    d3=pred2 - MatY.T
    
    g_prime2=pred1*(1-pred1)

    d2=np.dot(d3,Theta2)*g_prime2
    
    d2b = d2[:, 1:]
    
    for j in range(m) :
        delta2 = delta2 + np.outer((d3[j, :]).T,pred1[j, :])
        delta1 = delta1 + np.outer((d2b[j, :]).T,X1[j, :])

   # gradiant regularized
    
    Theta10 = np.copy(Theta1)
    np.put(Theta10,0,0)
    
    Theta20 = np.copy(Theta2)
    np.put(Theta20,0,0)
    


    Theta1_grad = delta1/m + (Lambda/m)*Theta10
    Theta2_grad = delta2/m + (Lambda/m)*Theta20
    

    
     # Unroll gradient
    grad = np.hstack((Theta1_grad.T.ravel(), Theta2_grad.T.ravel()))


    return J, grad
   

 
