import numpy as np
def featureNormalize(X):
#   Returns a normalized version where
#   mean value of each feature is 0 and the standard deviation is 1.
    mu = np.mean(X)
    sigma = np.std(X)
    a = X - mu
    X_norm = a / sigma
    return X_norm, mu, sigma