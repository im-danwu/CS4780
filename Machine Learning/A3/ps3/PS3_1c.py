from numpy import *
from numpy.linalg import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

S = ((array([1, 5, 1]), -1), (array([3, 5, 1]), -1),
    (array([4, 7, 1]), 1), (array([4, 9, 1]), 1),
    (array([6, 9, 1]), 1), (array([3, 1, 1]), -1))

w = array([0, 0, 0])
B = 0.5

def perceptron(S, w, B):
    check = True
    for i in range(6):
        weight = w[0:2]
        norm = math.sqrt(dot(weight, weight))
        # if x_i is incorrect, then adjust 
        if (dot(w, w)==0):
            w = w + S[i][1]*S[i][0]
        elif float(S[i][1]*dot(w, S[i][0]))/norm <= B*(math.sqrt(5)/2):
            w = w + S[i][1]*S[i][0]

    # check if x_i is on the right side
    for i in range(6):
        weight = w[0:2]
        norm = math.sqrt(dot(weight, weight))
        if (dot(w, w)!=0):
            if float(S[i][1]*dot(w, S[i][0]))/norm <= B*(math.sqrt(5)/2):
                check = False


    # if there was an element misclassified, rerun
    if check:
        return w
    else:
        return perceptron(S, w, B)

b = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

plt.plot(4, 7, 4, 9, 6, 9, marker='x', color='b')
plt.plot(1,5,3,5,3,1, marker='D', color='r')

x = linspace(0.0, 7.0, 500)
y = (-0.4*x + 6.2)*1./0.8
plt.plot(x, y, label='OP')

for B in b:
    perc = perceptron(S, w, B)
    x = linspace(0.0, 7.0, 500)
    y = (-perc[0]*x - perc[2])*1./perc[1]
    plt.plot(x, y, label=B)

# Now add the legend with some customizations
legend = plt.legend(bbox_to_anchor=(0,0,1.15,
    1),loc=1, shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')

plt.axis([0,7,0,10])
plt.show()
