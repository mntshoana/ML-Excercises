## Machine Learning - Exercise 1: Linear Regression (univariate)
#
# INITIALIZE
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
os.system('cls' if os.name == 'nt' else 'clear')
plt.close("all")
## =================== Part 2: Plotting   ====================
from python.plotData import plotData
# X represents population size in units of 10,000s
# y represents profit in units of $10,000s
print('Plotting data')
data = pd.read_csv('data/ex1data1.txt', names=['Population', 'Profit'])
X = data.iloc[:,0]
y = data.iloc[:,1]
#FILE: plotData.py
plotData(X,y)
input('Paused. Press enter to continue.\n')

## ============== Part 3: Cost and Gradient descent =============
from python.computeCost import computeCost
print('Testing the cost function ...')
m = data.shape #tuple (row, col)
m = m[0] # row
X = np.append(np.ones((m, 1)), data.iloc[:, 0].to_numpy().reshape(m,1), axis=1 )
y = data.iloc[:, 1].to_numpy().reshape(m,1)
theta = np.zeros((2,1))
J = computeCost(X, y, theta)
print(' - With theta = [0 ; 0]\nCost computed = ', J)
print(' - Expected cost value (approx) 32.07')
J = computeCost(X, y, np.c_[[-1 , 2]] )
print('\n - With theta = [-1 ; 2]\nCost computed = ', J)
print(' - Expected cost value (approx) 54.24')
input('Paused. Press enter to continue.\n')

from python.gradientDescent import gradientDescent
print('Running Gradient Descent ...')
iterations = 1500
alpha = 0.01
theta,Jhist = gradientDescent(X, y, theta, alpha, iterations)
print('- Theta found:\n')
print(theta)
print('- Expected theta values (approx)\n')
print('  -3.6303\n  1.1664\n\n')

# Plot the linear fit
plt.plot(range(1, m+1), X[:, 1], '-')
plt.plot(range(1, m+1), X.dot(theta), '-')
plt.legend(['Training data', 'Linear regression'])
plt.show()
#Predict
predict1 = np.array([1, 3.5]).dot(theta); # population of 35,000
print('For population = 35,000, we predict a profit of %f\n', predict1*10000)
predict2 = np.array([1, 7]).dot(theta); # population of 70,000
print('For population = 70,000, we predict a profit of %f\n', predict2*10000)
input('Paused. Press enter to continue.\n')
##=======================TODO PART 4 in python =======================
