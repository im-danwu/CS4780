import os
import math
from sklearn.datasets import load_svmlight_file
import scipy
from numpy import *
from numpy.linalg import *



# classify = subprocess.check_output(("./svm_classify testfile"+repr(m+1)+".dat"+"model"+repr(m+1)+".model"+"svm_predictions"+repr(m+1)),shell=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
test_file = os.path.join(current_dir, "boxes.test")

(XT, YT) = load_svmlight_file(test_file)

falsepos1 = 0
falsepos2 = 0
falsepos3 = 0
falseneg1 = 0
falseneg2 = 0
falseneg3 = 0

# for the predictions file with j = 0.5
with open('predictions_jj0.5.model') as fil:
    i = 0
    for line in fil:
        if YT[i] == -1:
            if line > 0.:
                falsepos1 = falsepos1 + 1
        if YT[i] == 1:
            if line <= 0.:
                falseneg1 = falseneg1 + 1
        i = i + 1

# for the predictions file with j = 0.1
with open('predictions_jj0.1.model') as fil:
    i = 0
    for line in fil:
        if YT[i] == -1:
            if line > 0.:
                falsepos2 = falsepos2 + 1
        if YT[i] == 1:
            if line <= 0.:
                falseneg2 = falseneg2 + 1
        i = i + 1

# for the predictions file with j = 0.05
with open('predictions_jj0.05.model') as fil:
    i = 0
    for line in fil:
        if YT[i] == -1:
            if line > 0.:
                falsepos3 = falsepos3 + 1
        if YT[i] == 1:
            if line <= 0.:
                falseneg3 = falseneg3 + 1
        i = i + 1


print "For j=0.5, number of false positives is %d and number of false negatives is %d." % (falsepos1, falseneg1)
print "For j=0.1, number of false positives is %d and number of false negatives is %d." % (falsepos2, falseneg2)
print "For j=0.05, number of false positives is %d and number of false negatives is %d." % (falsepos3, falseneg3)