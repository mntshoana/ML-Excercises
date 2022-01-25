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
from plotData import plotData
# X represents population size in units of 10,000s
# y represents profit in units of $10,000s
print('Plotting data')
data = pd.read_csv('../ex1data1.txt', names=['Population', 'Profit'])
X = data.iloc[:,0]
y = data.iloc[:,1]
#FILE: plotData.py
plotData(X,y)
input('Paused. Press enter to continue.\n')

## ============== Part 3: Cost and Gradient descent =============
from computeCost import computeCost
print('Testing the cost function ...')
m = data.shape #tuple (row, col)
m = m[0] # row
X = np.append(np.ones((m, 1)), data.iloc[:, 0].to_numpy().reshape(m,1), axis=1 )
y = data.iloc[:, 1].to_numpy().reshape(m,1)
theta = np.zeros((2,1))
J = computeCost(X, y, theta)
print(' - With theta = [0 ; 0]\nCost computed = ', J)
print(' - Expected cost value (approx) 32.07')
##=======================TODO PART 4 in python =======================
