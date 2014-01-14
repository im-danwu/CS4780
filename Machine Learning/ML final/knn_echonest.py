from __future__ import division
import svmlight_loader as svm
import os
import math
import operator
import numpy as np
import matplotlib.pyplot as mplt

# Number of nearest neighbors to use
k = 30
assert k <= 300
labels = ['electronic', 'metal', 'rap', 'classical']

# Load SVMLight files as numpy arrays
currdir = os.path.dirname(os.path.abspath(__file__))
trainfile = os.path.join(currdir, "data", "songsv1.train")
testfile = os.path.join(currdir, "data", "songsv1.test.txt")
x_train, y_train = svm.load_svmlight_file(trainfile)
x_test, y_test = svm.load_svmlight_file(testfile)

# Convert sparse matrix to dense matrix
x_train = x_train.todense()
x_test = x_test.todense()

x_train = np.array(x_train)
x_test = np.array(x_test)

# Statistics
num_correct = 0
num_correct_electronic = 0
num_correct_metal = 0
num_correct_rap = 0
num_correct_classical = 0
num_predict_electronic = 0
num_predict_metal = 0
num_predict_rap = 0
num_predict_classical = 0
num_actual_electronic = 0
num_actual_metal = 0
num_actual_rap = 0
num_actual_classical = 0
num_total = 0

# For each test song...
for i in range(len(x_test)):
    knn = {'idx':'sim_measure'}
    test_song = x_test[i]
    # For each training example...
    for m in range(len(x_train)):
        train_song = x_train[m]
        # Compute euclidean distance between feature vectors
        sum = 0
        for j in range(len(train_song) - 1):
            # Ignore certain features
            if j == 1 or j == 2 or j == 3 or j == 7 or j == 8: continue
            sum += math.pow((test_song[j] - train_song[j]), 2)
        euc_dist = math.sqrt(sum)
        knn[m] = euc_dist
    # Sort dict of distances
    sorted_knn = sorted(knn.iteritems(), key=operator.itemgetter(1))
    # Determine number of votes for each class
    votes = [0., 0., 0., 0.]
    for n in range(k):
        label = y_train[sorted_knn[n][0]]
        votes[int(label) - 1] += (k - n) / k
    # Classify based on max number of votes
    classification = votes.index(max(votes))
    
    # Analytics
    if (classification + 1) == y_test[i]:
        num_correct += 1
        if classification == 0:
            num_correct_electronic += 1
        if classification == 1:
            num_correct_metal += 1
        if classification == 2:
            num_correct_rap += 1
        if classification == 3:
            num_correct_classical += 1
    else:
        if classification == 0:
            num_predict_electronic += 1
        if classification == 1:
            num_predict_metal += 1
        if classification == 2:
            num_predict_rap += 1
        if classification == 3:
            num_predict_classical += 1
        if y_test[i] == 1:
            num_actual_electronic += 1
        if y_test[i] == 2:
            num_actual_metal += 1
        if y_test[i] == 3:
            num_actual_rap += 1
        if y_test[i] == 4:
            num_actual_classical += 1
    num_total += 1
    
    #print "Number of votes for electronic: " + str(votes[0])
    #print "Number of votes for metal: " + str(votes[1])
    #print "Number of votes for rap: " + str(votes[2])
    #print "Number of votes for classical: " + str(votes[3])
    #print "Actual classification: " + labels[int(y_test[i])-1]

print "Predicted electronic: " + str(num_predict_electronic)
print "Actual electronic: " + str(num_actual_electronic)
print "Predicted metal: " + str(num_predict_metal)
print "Actual metal: " + str(num_actual_metal)
print "Predicted rap: " + str(num_predict_rap)
print "Actual rap: " + str(num_actual_rap)
print "Predicted classical: " + str(num_predict_classical)
print "Actual classical: " + str(num_actual_classical)
print "Correct electronic: " + str(num_correct_electronic)
print "Correct metal: " + str(num_correct_metal)
print "Correct rap: " + str(num_correct_rap)
print "Correct classical: " + str(num_correct_classical)
print "Correct predictions: " + str(num_correct)
print "Total predictions: " + str(num_total)
print "Total accuracy: " + str(num_correct / num_total)

