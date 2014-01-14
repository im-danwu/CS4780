import os
import math
from sklearn.datasets import load_svmlight_file
import scipy
from numpy import *
from numpy.linalg import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# execfile(svm2weight["model100_1.model"])
current_dir = os.path.dirname(os.path.abspath(__file__))
training_file = os.path.join(current_dir, "boxes.train")

(Xtr, Ytr) = load_svmlight_file(training_file)
Xtr = Xtr.todense()

#model100_1, for nneg = 100 and C = 1
w1a = -3.69928425158
w2a = -2.94525363682
b1 = -3.338982
xa = linspace(0.0, 1.0, 500)
ya = (-w1a*xa + b1)/w2a

plt.figure("nneg = 100 and C = 1")
plt.plot(xa, ya)
for i in range(200):
    if Ytr[i] == 1:
        plt.plot(Xtr[i,0], Xtr[i,1], marker='+', color='r')
    else:
        plt.plot(Xtr[i,0], Xtr[i,1], marker='o', color='b')
plt.axis([0,1,0,1])
plt.show()

#model100_1000, for nneg = 100 and C = 1000
w1b = -10.2401000648
w2b = -12.2027628803
b2 = -10.753711
xb = linspace(0.0, 1.0, 500)
yb = (-w1b*xb + b2)/w2b

plt.figure("nneg = 100 and C = 1000")
plt.plot(xb, yb)
for i in range(200):
    if Ytr[i] == 1:
        plt.plot(Xtr[i,0], Xtr[i,1], marker='+', color='r')
    else:
        plt.plot(Xtr[i,0], Xtr[i,1], marker='o', color='b')
plt.axis([0,1,0,1])
plt.show()

#model50_1, for nneg = 50 and C = 1
w1c = -3.29902116702
w2c = -2.82044298227
b3 = -3.2203815
xc = linspace(0.0, 1.0, 500)
yc = (-w1c*xc + b3)/w2c

plt.figure("nneg = 50 and C = 1")
plt.plot(xc, yc)
for i in range(150):
    if Ytr[i] == 1:
        plt.plot(Xtr[i,0], Xtr[i,1], marker='+', color='r')
    else:
        plt.plot(Xtr[i,0], Xtr[i,1], marker='o', color='b')
plt.axis([0,1,0,1])
plt.show()

#model50_1000, for nneg = 50 and C = 1000
w1d = -10.2401000648
w2d = -12.2027628803
b4 = -10.753711
xd = linspace(0.0, 1.0, 500)
yd = (-w1d*xd + b4)/w2d

plt.figure("nneg = 50 and C = 1000")
plt.plot(xd, yd)
for i in range(150):
    if Ytr[i] == 1:
        plt.plot(Xtr[i,0], Xtr[i,1], marker='+', color='r')
    else:
        plt.plot(Xtr[i,0], Xtr[i,1], marker='o', color='b')
plt.axis([0,1,0,1])
plt.show()

#model10_1, for nneg = 10 and C = 1
w1e = -2.14830082406
w2e = -2.31499977484
b5 = -2.7738797
xe = linspace(0.0, 1.0, 500)
ye = (-w1e*xe + b5)/w2e

plt.figure("nneg = 10 and C = 1")
plt.plot(xe, ye)
for i in range(110):
    if Ytr[i] == 1:
        plt.plot(Xtr[i,0], Xtr[i,1], marker='+', color='r')
    else:
        plt.plot(Xtr[i,0], Xtr[i,1], marker='o', color='b')
plt.axis([0,1,0,1])
plt.show()

#model10_1000, for nneg = 10 and C = 1000
w1f = -10.2401000648
w2f = -12.2027628803
b6 = -10.753711
xf = linspace(0.0, 1.0, 500)
yf = (-w1f*xf + b6)/w2f

plt.figure("nneg = 10 and C = 1000")
plt.plot(xf, yf)
for i in range(110):
    if Ytr[i] == 1:
        plt.plot(Xtr[i,0], Xtr[i,1], marker='+', color='r')
    else:
        plt.plot(Xtr[i,0], Xtr[i,1], marker='o', color='b')
plt.axis([0,1,0,1])
plt.show()


