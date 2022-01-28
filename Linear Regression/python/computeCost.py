import numpy as np
def computeCost(X, y, theta):
#   Cost for linear regression
    h = X.dot( theta )
    a = (h - y).T
    b = h - y

    m = h.shape[0] # Number of trainning examples
    mean = 1 / (2 * m)

    J = mean * (a.dot( b ))
    return J
