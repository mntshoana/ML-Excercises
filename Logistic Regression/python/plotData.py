import matplotlib.pyplot as plt
def plotData(X, y):
#  X is assumed to be a Mx2 matrix.
    plt.figure

    pos = X[ y == 1 ]
    neg = X[ y == 0 ]
    
    plt.plot(pos.iloc[:, 0], pos.iloc[:, 1], 'k+', linewidth=2, markersize=7)
    plt.plot(neg.iloc[:, 0], neg.iloc[:, 1], 'ko', markerfacecolor='y', markersize=7)
    return