import numpy as np
import operator
#http://matplotlib.org/users/image_tutorial.html
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math as m
import scipy.sparse as sps
from scipy import sparse
from scipy.spatial import distance
from sklearn.datasets import load_svmlight_file
# load dataset: X_Train is a sparce matrix (each line is a person
# Y_train is empty (usually is the correction stating +1 or -1 tec)
#scipy sparse CSR matrices
Xtrain_matrix, Ytrain_matrix = load_svmlight_file("./books.train")
Xtest_matrix, Ytest_matrix = load_svmlight_file("./books.test")

train_matrix = Xtrain_matrix.todense()
test_matrix = Xtest_matrix.todense()

k = 5

#outputs a class label c for every instance
sizeTestSet = 5000
correctPredictions = 0
test_row = 0
for test in test_matrix:
  classTotals = {0.0: 0, 1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0}
  scores = {}
  i = 0
  for train in train_matrix:
    d = (test*train_matrix[i].transpose()).sum()
    e = np.linalg.norm(test)
    f = np.linalg.norm(train_matrix[i])
    if ((e*f) == 0):
      scores[i] = 0
    else:
      scores[i] = d/(e*f)
    i+= 1
  sorted_scores = sorted(scores.iteritems(), key=operator.itemgetter(1), 
  reverse=True)
  sorted_scores = sorted_scores[:k]
  for neighbor in sorted_scores:
    row_num = neighbor[0] #actual row num is +1 in notepad
    label = Ytrain_matrix[row_num]
    classTotals[label]+= 1
  prediction = max(classTotals.iteritems(), key=operator.itemgetter(1))[0]
  if (Ytest_matrix[test_row] == prediction):
    correctPredictions+= 1
  test_row+= 1

accuracy = correctPredictions/float(sizeTestSet)
print "Accuracy " + str(accuracy)
#TODO: do something with log(k)
#k=1 -> accuracy is 
#k=2 -> accuracy is 
#k=5 -> accuracy is
#k=10 ->
#k=100 ->
#k=200 ->
#k=300 ->
#k=500 ->
#k=1000 ->
#k=2000 ->
#k=3000 ->
#k=4000 ->
#k=5000 ->
