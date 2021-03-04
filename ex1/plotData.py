import matplotlib.pyplot as plt
def plotData(x, y):
#   Plot data points
#   x = population, y = profit
    plt.figure
    plt.plot(x, y, 'rx', markersize=10)
    plt.ylabel('Profit in $10,000s');
    plt.xlabel('Population of City in 10,000s');
    plt.show(block=False)
    return

