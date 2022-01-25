import numpy as np
def computeCost(X, y, theta):
#   Cost for linear regression
    print("X: ", X.shape)
    print("y: ", y.shape)
    h = X.dot( theta )
    a = (h - y).T
    b = h - y
    print("a: ", a.shape)
    print("b: ", b.shape)
    m = h.shape[0] # Number of trainning examples
    mean = 1 / (2 * m)
    print("mean: ", mean)
    J = mean * (a.dot( b ))
    return J
