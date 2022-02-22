import numpy as np
from python.sigmoid import sigmoid

def computeCost(X, y, theta):
#   Logistic regression
#   Cost and gradient 
    m = y.size

#   HYPOTHESIS
    z = X.dot( theta )
    h = sigmoid(z)

    # COST
    a = - y.T.dot( np.log(h) )
    b = (1 - y).T.dot( np.log(1 - h) )
    J = 1/m * (a - b)

    # GRADIENT
    deltaJ = 1/m * X.T.dot( ( h - y ) )
    return J, deltaJ