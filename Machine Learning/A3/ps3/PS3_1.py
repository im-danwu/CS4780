from numpy import *
from numpy.linalg import *

S = ((array([1, 5, 1]), 1), (array([3, 5, 1]), 1),
    (array([4, 7, 1]), -1), (array([4, 9, 1]), -1),
    (array([6, 9, 1]), -1), (array([3, 1, 1]), 1))

w = array([0, 0, 0])

def perceptron(S, w):
    check = True
    for i in range(6):
        # if x_i is incorrect, then adjust 
        if S[i][1]*dot(w, S[i][0]) <= 0:
            w = w + S[i][1]*S[i][0]

    # check if x_i is on the right side
    for i in range(6):
        if (dot(w, S[i][0]) > 0) & (S[i][1] == -1):
            check = False
        elif (dot(w, S[i][0]) <= 0) & (S[i][1] == 1):
            check = False

    # if there was an element misclassified, rerun
    if check:
        print "done"
        print w
        return w
    else:
        print "rerun"
        print w
        perceptron(S, w)


perceptron(S, w)