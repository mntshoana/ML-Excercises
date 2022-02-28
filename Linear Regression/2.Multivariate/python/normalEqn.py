import numpy as np
def normalEqn(X, y):
#  Computes linear regression using the normal equations.
#  Normal Equation is an analytical approach to Linear Regression with a Least Square Cost Function
    colSize = X.shape[1]
    theta = np.zeros((colSize , 1))

    a = np.linalg.pinv( X.T.dot( X ) )
    b = X.T
    theta = a.dot( b).dot( y )
    return theta
