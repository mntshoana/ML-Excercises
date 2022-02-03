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
print('Running gradient descent ...\n')

# Choose some alpha value
alpha = 0.01
num_iters = 400

# Init Theta and Run Gradient Descent 
y = y.to_numpy().reshape(m,1)
theta = np.zeros((3, 1))
from python.gradientDescent import gradientDescent
theta, J_history = gradientDescent(Xnorm, y, theta, alpha, num_iters)

# Plot the convergence graph
# Cost
plt.plot(range(0, J_history.size), J_history, '-b', linewidth=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

# Theta after gradient descent
print('Theta computed from gradient descent:\n', theta)

# Estimate the price of a 1650 sq-ft, 3 br house
price = np.array([1, (1650 - mu)/sigma , (3 - mu)/sigma]).dot(theta)
print('Predicted price of a 1650 sq-ft, 3 bedroom house', 
    '(using gradient descent):\n', price)

input('Paused. Press enter to continue.\n')

# ================ Normal Equations ================
# Still requires intercept term (Theta 0)
print("Using normal equation...\n")
X = np.append(np.ones((m, 1)), X.to_numpy().reshape(m,2), axis=1 )
from python.normalEqn import normalEqn
theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations: \n', theta, '\n')

# Estimate the price of a 1650 sq-ft, 3 br house
price = np.array([1, 1650  , 3 ]).dot(theta)
print('Predicted price of a 1650 sq-ft, 3 br house ',
         '(using normal equations):\n$', price[0])
