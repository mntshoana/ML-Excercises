## Machine Learning - Exercise 1: Linear Regression
#
# INITIALIZE
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
os.system('cls' if os.name == 'nt' else 'clear')
plt.close("all")
## =================== Part 1: Basic Function ===================
from warmUpExercise import warmUpExercise
print('A basic function.')
print(' - A 5x5 Identity Matrix: \n');
# FILE: warmUpExercise.py
print(warmUpExercise(), '\n')
input('Paused. Press enter to continue.\n');

## ===================    Part 2: Plotting   ====================
from plotData import plotData
# x= population size in 10,000s
# y= profit in $10,000s
print('Plotting data')
data = pd.read_csv('ex1data1.txt', names=['Population', 'Profit'])
X = data.iloc[:,0]
y = data.iloc[:,1]
m = data.shape[0] #tuple (row, col) // looking for row
#FILE: plotData.py
plotData(X,y)
input('Paused. Press enter to continue.\n');

## ============== Part 3: Cost and Gradient descent =============
from computeCost import computeCost
print('Testing the cost function ...')
X = np.c_[np.ones((m,1)), data.iloc[:,0]]; #MX2
y = np.c_[y];
theta = np.zeros((2,1));                    #2X1
J = computeCost(X, y, theta)
print(' - With theta = [0 ; 0]\nCost computed = ', J);
print(' - Expected cost value (approx) 32.07');
J = computeCost(X, y, np.c_[[-1 , 2]] );
print('\n - With theta = [-1 ; 2]\nCost computed = ', J);
print(' - Expected cost value (approx) 54.24');
input('Paused. Press enter to continue.\n');

