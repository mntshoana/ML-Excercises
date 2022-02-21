## Logistic Regression
#
# INITIALIZE
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
os.system('cls' if os.name == 'nt' else 'clear')
plt.close("all")

data = pd.read_csv('data/ex2data1.txt')
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
# TODO