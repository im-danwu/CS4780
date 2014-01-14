import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import math

#plt.plot(4, 7, 4, 9, 6, 9, marker='x', color='b')
#plt.plot(1,5,3,5,3,1, marker='D', color='r')

#plt.axis([0,10,0,10])
#C = [math.log(0.001,2), math.log(0.01,2), math.log(0.1,2), math.log(0.5,2), math.log(1.0,2), math.log(2.0,2), math.log(5.0,2), math.log(10.0,2), math.log(100.0,2)]
C = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]

#x = np.arange(0, math.log(100.0, 2), );
#y = np.sin(x)
#plt.plot(y)
#plt.show()

from matplotlib import pyplot
import math
#pyplot.plot([x for x in range(100)],[y for y in range(100)] )
#pyplot.xlabel('log2 C')
#pyplot.ylabel('Accuracy (%)')
#pyplot.title('Test Accuracies')
#
#pyplot.xscale('log')
#pyplot.yscale('log')


pyplot.show()