## Univariate Linear Regression
#
# INITIALIZE
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
os.system('cls' if os.name == 'nt' else 'clear')
plt.close("all")
## ===================== Plotting   ===================
# X represents population size in units of 10,000s
# y represents profit in units of $10,000s
print('Plotting data')
data = pd.read_csv('data/ex1data1.txt', names=['Population', 'Profit'])
X = data.iloc[:,0]
y = data.iloc[:,1]
#FILE: plotData.py
from python.plotData import plotData
plotData(X,y)
input('Paused. Press enter to continue.\n')

## ============== Part 3: Cost and Gradient descent =============
print('Testing the cost function ...')
m = data.shape #tuple (row, col)
m = m[0] # row
X = np.append(np.ones((m, 1)), data.iloc[:, 0].to_numpy().reshape(m,1), axis=1 )
y = data.iloc[:, 1].to_numpy().reshape(m,1)
theta = np.zeros((2,1))
from python.computeCost import computeCost
J = computeCost(X, y, theta)
print(' - With theta = [0 ; 0]\nCost computed = ', J)
print(' - Expected cost value (approx) 32.07')
J = computeCost(X, y, np.c_[[-1 , 2]] )
print('\n - With theta = [-1 ; 2]\nCost computed = ', J)
print(' - Expected cost value (approx) 54.24')
input('Paused. Press enter to continue.\n')


print('Running Gradient Descent ...')
iterations = 1500
alpha = 0.01
from python.gradientDescent import gradientDescent
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
## =========== Visualizing J(theta_0, theta_1) ==========
print('Visualizing J(theta_0, theta_1) ...')

# Grid
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
np.zeros((2,1))
J_vals = np.zeros((theta0_vals.size, theta1_vals.size))

for i in range(0, theta0_vals.size):
    for j in range(0, theta1_vals.size):
        t = np.c_[[theta0_vals[i], theta1_vals[j]]]
        J_vals[i,j] = computeCost(X, y, t)
    
# With the surf command, we'll need to transpose J_vals
# or else the axes will be flipped
J_vals = J_vals.T

# Surface plot
ax = plt.axes(projection='3d')
a, b = np.meshgrid(theta0_vals, theta1_vals)
ax.plot_surface(a, b, J_vals)
ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')

plt.show()

# Contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)
plt.show()
input('Paused. Press enter to continue.\n')
