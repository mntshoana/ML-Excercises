import numpy
from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):
#   Performs gradient descent and updates theta using the learning rate alpha
    m = y.size; # number of training examples
    J_history = numpy.zeros((num_iters, 1))

    for iter in range(1, num_iters):
        c = 1/m
        h = X.dot( theta )
        DeltaJ =  c * (X.T).dot(h - y)
        theta = theta - alpha * DeltaJ

        # Save the cost J in every iteration
        J_history[iter] = computeCost(X, y, theta)
    return theta, J_history
