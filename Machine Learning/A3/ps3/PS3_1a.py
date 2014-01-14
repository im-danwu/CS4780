from numpy import *
from numpy.linalg import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

S = ((array([1, 5, 1]), -1), (array([3, 5, 1]), -1),
    (array([4, 7, 1]), 1), (array([4, 9, 1]), 1),
    (array([6, 9, 1]), 1), (array([3, 1, 1]), -1))

w = array([0, 0, 0])

def perceptron(S, w):
    check = True
    for i in range(6):
        # if x_i is incorrect, then adjust 
        if S[i][1]*dot(w, S[i][0]) <= 0:
            w = w + S[i][1]*S[i][0]

    # check if x_i is on the right side
    for i in range(6):
        if (S[i][1]*dot(w, S[i][0]) <= 0):
            check = False

    # if there was an element misclassified, rerun
    if check:
        return w
    else:
        return perceptron(S, w)


perc = perceptron(S, w)
print perc

x = linspace(0.0, 7.0, 1000)
y = (-perc[0]*x - perc[2])*1./perc[1]

plt.plot(x, y)
plt.plot(4, 7, 4, 9, 6, 9, marker='x', color='b')
plt.plot(1,5,3,5,3,1, marker='D', color='r')

plt.axis([0,7,0,10])
plt.show()




