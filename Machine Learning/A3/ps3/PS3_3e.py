import os
import svm2weight
import math
from sklearn.datasets import load_svmlight_file
import scipy
from numpy import *
from numpy.linalg import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

current_dir = os.path.dirname(os.path.abspath(__file__))
training_file = os.path.join(current_dir, "boxes.train")

(Xtr, Ytr) = load_svmlight_file(training_file)
Xtr = Xtr.todense()

#modelj_0.5, for nneg = 10, C = 1, j = 0.5
w1a = -2.07038099469
w2a = -2.12657491253
b1 = -2.5445036
xa = linspace(0.0, 1.0, 500)
ya = (-w1a*xa + b1)/w2a

plt.figure("nneg = 10, C = 1, and j = 0.5")
plt.plot(xa, ya)
for i in range(110):
    if Ytr[i] == 1:
        plt.plot(Xtr[i,0], Xtr[i,1], marker='+', color='r')
    else:
        plt.plot(Xtr[i,0], Xtr[i,1], marker='o', color='b')
plt.axis([0,1,0,1])
plt.show()

#modelj_0.5, for nneg = 10, C = 1, j = 0.5
w1b = -1.95125798849
w2b = -1.80882055295
b2 = -1.8616103 
xb = linspace(0.0, 1.0, 500)
yb = (-w1b*xb + b2)/w2b

plt.figure("nneg = 10, C = 1, and j = 0.1")
plt.plot(xb, yb)
for i in range(110):
    if Ytr[i] == 1:
        plt.plot(Xtr[i,0], Xtr[i,1], marker='+', color='r')
    else:
        plt.plot(Xtr[i,0], Xtr[i,1], marker='o', color='b')
plt.axis([0,1,0,1])
plt.show()

#modelj_0.5, for nneg = 10, C = 1, j = 0.05
w1c = -1.6279945727
w2c = -1.52789447603
b3 = -1.3969913 
xc = linspace(0.0, 1.0, 500)
yc = (-w1c*xc + b3)/w2c

plt.figure("nneg = 10, C = 1, and j = 0.05")
plt.plot(xc, yc)
for i in range(110):
    if Ytr[i] == 1:
        plt.plot(Xtr[i,0], Xtr[i,1], marker='+', color='r')
    else:
        plt.plot(Xtr[i,0], Xtr[i,1], marker='o', color='b')
plt.axis([0,1,0,1])
plt.show()




