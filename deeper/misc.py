##
# Miscellaneous helper functions
##

import numpy as np
from numpy.random import uniform

def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####
    epsilon = np.sqrt(6.0 / (m + n))
    A0 = uniform(low=-epsilon, high=epsilon, size=(n * m))
    A0 = A0.reshape((m,n))
    #### END YOUR CODE ####
    assert(A0.shape == (m,n))
    return A0