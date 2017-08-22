import numpy as np


def computeCentroids(X, idx, K):
    """returns the new centroids by
    computing the means of the data points assigned to each centroid. It is
    given a dataset X where each row is a single data point, a vector
    idx of centroid assignments (i.e. each entry in range [1..K]) for each
    example, and K, the number of centroids. You should return a matrix
    centroids, where each row of centroids is the mean of the data points
    assigned to it.
    """

# Useful variables
    m, n = X.shape

# You need to return the following variables correctly.
    centroids = []
    #centroids=np.zeros((K,n))
    for k in range(K):
        ss_ens=X[idx==k,:]
        centroids.append(np.mean(ss_ens, axis=0))
   

# ====================== YOUR CODE HERE ======================
# Instructions: Go over every centroid and compute mean of all points that
#               belong to it. Concretely, the row vector centroids(i, :)
#               should contain the mean of the data points assigned to
#               centroid i.
#
# Note: You can use a for-loop over the centroids to compute this.
# 


# =============================================================
    #Z=np.column_stack((X,idx)) 
    #Z[np.where(Z[:,2]==1),:]
    #s=np.zeros(K)
    #for i in range(m) :
     #   centroids[idx[i]-1,:]= centroids[idx[i]-1,:]+ X[i,:]
      #  s[idx[i]-1]=1+s[idx[i]-1]
    
    #centroids=centroids/s
     
    #### autre maniere
        
        
 #   for i in range(m):
  #      for k in range(K):
   #         if (idx[i]==k):
    #            s[k]=s[k]+1
     #           centroids[k,0]=centroids[k,0]+X[i,0]
      #          centroids[k,1]=centroids[k,1]+X[i,1]
        
 #   centroids[:,0]=centroids[:,0]/s
  #  centroids[:,1]=centroids[:,1]/s
        
 #  autre maniere :


    #s=np.zeros(K) 
#    for nb in range(n):   
#        for i in range(m):
 #           for k in range(K):
              
     
            
   #             if (idx[i]==k):
   #                 s[k]=s[k]+1
                   
    #                centroids[k,nb]=centroids[k,nb]+X[i,nb]
                    
         
     #   centroids[:,nb]=centroids[:,nb]/s
  

    
    return centroids
