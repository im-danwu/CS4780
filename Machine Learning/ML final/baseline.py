from __future__ import division
from sklearn.datasets import load_svmlight_file
import os
import math
import operator
import numpy as np
import random

NUM_CLASSES = 4

# Load SVMLight files as numpy arrays
currdir = os.path.dirname(os.path.abspath(__file__))
trainfile = os.path.join(currdir, "data", "songsv1.train")
testfile = os.path.join(currdir, "data", "songsv1.test.txt")
x_train, y_train = load_svmlight_file(trainfile)
x_test, y_test = load_svmlight_file(testfile)

# Convert sparse matrix to dense matrix
x_train = x_train.todense()
x_test = x_test.todense()

x_train = np.array(x_train)
x_test = np.array(x_test)


size    = len(y_test)
counts  = [0 for i in range(NUM_CLASSES)]
guesses = [0 for i in range(NUM_CLASSES)]
errors  = [0 for i in range(NUM_CLASSES)]
correct = [0 for i in range(NUM_CLASSES)]

for i in y_test:
    counts[i-1] = counts[i-1] + 1
    g = random.randint(1,NUM_CLASSES)
    guesses[g-1] = guesses[g-1] + 1
    if i != g:
        errors[i-1] = errors[i-1] + 1
    elif i == g:
        correct[i-1] = correct[i-1] + 1

#error and accuracy
total_errors = 0
for e in errors:
    total_errors = total_errors + e
error_rate = float(total_errors/size)
accuracy   = 1-error_rate

#precisions and recalls
precisions = [0 for i in range(NUM_CLASSES)]
recalls    = [0 for i in range(NUM_CLASSES)]
for j in range(NUM_CLASSES):
    precisions[j] = float(correct[j]/guesses[j])
    recalls[j]    = float(correct[j]/counts[j])

#writing to file
with open('baseline_results', 'w') as f:
    f.write("error rate was "+str(error_rate))
    f.write("accuracy was "+str(accuracy))
    for i in range(NUM_CLASSES):
        f.write("precision for class %d was: %.2f") %(i+1, precisions[i])
        f.write("recall for class %d was: %.2f") %(i+1, recalls[i])







# # Count class distribution of training set
# size = len(y_train)
# totals = [0]*NUM_CLASSES

# for i in range(size):
#     genre = y_train[i]
#     totals[genre-1] = totals[genre-1] + 1

# # Probability of each class appearing based on the training set
# prob = [0]*NUM_CLASSES

# for j in range(NUM_CLASSES):
#     prob[j] = float(totals[j]/size)
