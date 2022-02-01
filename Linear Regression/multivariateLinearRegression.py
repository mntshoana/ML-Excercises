## Multivariate Linear Regression
#
# INITIALIZE
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
os.system('cls' if os.name == 'nt' else 'clear')
plt.close("all")
# ================ Feature Normalization ================
# X1 represents the size of a housen in sq-ft
# X2 represents the number of bedrooms
# y represents price of the house
data = pd.read_csv('data/ex1data2.txt', names=['Size', 'Bedrooms', 'Price'])
X = data.iloc[:, 0:2]
y = data.iloc[:, 2]
m = y.size

print('First 10 examples from the dataset: \n')
for val in range (0,10):
    print( (' x = [{0}, {1}], y = {2}'.format(X.iloc[val,0], X.iloc[val,1], y.iloc[val])))

# Scale features and set them to zero mean
print('Normalizing Features ...\n')
from python.featureNormalize import featureNormalize
Xnorm, mu, sigma = featureNormalize(X)

# Add intercept term to X
Xnorm = np.append(np.ones((m, 1)), Xnorm.to_numpy().reshape(m,2), axis=1 )
input('Paused. Press enter to continue.\n')

## ================ Gradient Descent ================
#print('Running gradient descent ...\n')
#Todo
