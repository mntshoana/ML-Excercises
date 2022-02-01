import numpy as np
def featureNormalize(X):
#   Returns a normalized version where
#   mean value of each feature is 0 and the standard deviation is 1.
    mu = np.mean(X)
    print(mu)
    sigma = np.std(X)
    print(sigma)
    a = X - mu
    print(a.head())
    X_norm = a / sigma
    print(X_norm.head())
    return X_norm, mu, sigma