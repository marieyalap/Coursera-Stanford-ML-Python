import numpy as np
from sklearn import svm


def dataset3Params(X, y, Xval, yval):
    """returns your choice of C and sigma. You should complete
    this function to return the optimal C and sigma based on a
    cross-validation set.
    """

# You need to return the following variables correctly.
    #C = 1
    #sigma = 0.3

# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return the optimal C and sigma
#               learning parameters found using the cross validation set.
#               You can use svmPredict to predict the labels on the cross
#               validation set. For example, 
#                   predictions = svmPredict(model, Xval)
#               will return the predictions on the cross validation set.
#
#  Note: You can compute the prediction error using 
#        mean(double(predictions ~= yval))
#


# =========================================================================
    vector_C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    vector_sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    best_error=10000000000
    C=0
    sigma=0

    for i in vector_C :
        for j in vector_sigma :
            gamma = 1.0 / (2.0 * (j ** 2))
            clf = svm.SVC(C=i, kernel='rbf', tol=1e-3, max_iter=200, gamma=gamma)
            model = clf.fit(X, y)
            predictions = model.predict(Xval)
            error = np.mean(predictions != yval)
            if (error<best_error) :
                best_error = error
                C=i
                sigma=j

    return C, sigma
