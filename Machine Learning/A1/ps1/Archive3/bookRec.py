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

k = 10
train_matrix = Xtrain_matrix.todense()
test_matrix = Xtest_matrix.todense()

#select a k
k = 10
greyTest = 2165
zombieTest = 3539 

#for 50 shades of grey
print "For Fifty Shades of Grey"
grey_matrix = train_matrix[greyTest]
scores = {}
for i in range(5000):
        d = (train_matrix[greyTest]*train_matrix[i].transpose()).sum()
        e = np.linalg.norm(grey_matrix) 
        f = np.linalg.norm(train_matrix[i])
        if ((e*f) == 0):
                scores[(i+1)] = 0
        else:
                scores[(i+1)] = d/(e*f)
sorted_scores = sorted(scores.iteritems(), key=operator.itemgetter(1), 
reverse=True)
print sorted_scores[1:(k+1)]


print "Brains: A Zombie Memoir"
zombie_matrix = train_matrix[zombieTest]
scores = {}
for i in range(5000):
        d = (train_matrix[zombieTest]*train_matrix[i].transpose()).sum()
        e = np.linalg.norm(zombie_matrix) 
        f = np.linalg.norm(train_matrix[i])
        if ((e*f) == 0):
                scores[(i+1)] = 0
        else:
                scores[(i+1)] = d/(e*f)
sorted_scores = sorted(scores.iteritems(), key=operator.itemgetter(1), 
reverse=True)
print sorted_scores[1:(k+1)]

