import numpy as np
from math import exp

def gaussianKernel(x1, x2, sigma):
    
    """returns a gaussian kernel between x1 and x2
    and returns the value in sim
    """

# Ensure that x1 and x2 are column vectors
#     x1 = x1.ravel()
#     x2 = x2.ravel()

# You need to return the following variables correctly.
    sim = 0
    v=0
    u = np.zeros(x1.size)
# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return the similarity between x1
#               and x2 computed using a Gaussian kernel with bandwidth
#               sigma
#
#

  #autre maniere avec le produit scalaire
    #v= np.dot((x1-x2).T,(x1-x2))
    #sim = exp(-v/(2*(sigma**2)))

# =============================================================
    x1=np.ravel(x1)
    x2=np.ravel(x2)

    for i in range(x1.size):   
        u[i]= (x1[i]-x2[i])**2
        v=v+u[i]
    sim = exp(-v/(2*(sigma**2)))

    return sim