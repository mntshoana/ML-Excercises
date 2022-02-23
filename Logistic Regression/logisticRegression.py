## Logistic Regression
#
# INITIALIZE
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
os.system('cls' if os.name == 'nt' else 'clear')
plt.close("all")

data = pd.read_csv('data/ex2data1.txt', names=['Exam 1 Score', 'Exam 2 Score', 'Admitted'])
X = data.iloc[:, 0:2]
y = data.iloc[:, 2]

# ================ Plotting ================
print(['Plotting data with + indicating (y = 1)',
        ' and o indicating (y = 0).\n'])

from python.plotData import plotData
plotData(X, y)

plt.ylabel('Exam 2 score')
plt.xlabel('Exam 1 score')

plt.show(block=False)

plt.legend(['Admitted', 'Not Admitted'])
input('Paused. Press enter to continue.\n')

# ============== Compute Cost and Gradient ==============
m, n = X.shape
X = np.append( np.ones((m, 1)), X.to_numpy().reshape(X.shape), axis=1)

from python.costFunction import costFunction
# Compute and display initial cost and gradient
initial_theta = np.zeros((n + 1, 1))
cost, grad = costFunction(X, y.to_numpy().reshape(y.size, 1), initial_theta)
print('Cost at initial theta (zeros): \n', cost)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros): \n', grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.c_[[-24, 0.2, 0.2]]
cost, grad = costFunction(X, y.to_numpy().reshape(y.size, 1), test_theta)
print('\nCost at test theta: ', cost)
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n', grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

input('Paused. Press enter to continue.\n')

# ============= Optimizing using fminunc  =============
# TODO