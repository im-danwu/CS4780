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

#for each class... split data into genres
#0 = rows 0 to 999
#2 = rows 1000 to 1999
#3 = rows 2000 to 2999
#1 = rows 3000 to 3999
#4 = rows 4000 to 4999

genres = {}

genre0 = [[0]*10000]
for i in range(999): 
        norm = np.linalg.norm(train_matrix[i])
        if (norm != 0):
                instance_vec = train_matrix[i]/norm
                genre0 = genre0 + instance_vec
genres[0] = genre0

genre2 = [[0]*10000]
for i in range(1000, 1999):
        norm = np.linalg.norm(train_matrix[i])
        if (norm != 0): 
                instance_vec = train_matrix[i]/norm
                genre2 = genre2 + instance_vec
genres[2] = genre2

genre3 = [[0]*10000]
for i in range(2000, 2999):
        norm = np.linalg.norm(train_matrix[i])
        if (norm != 0): 
                instance_vec = train_matrix[i]/norm
                genre3 = genre3 + instance_vec
genres[3] = genre3

genre1 = [[0]*10000]
for i in range(3000, 3999):
        norm = np.linalg.norm(train_matrix[i])
        if (norm != 0): 
                instance_vec = train_matrix[i]/norm
                genre1 = genre1 + instance_vec
genres[1] = genre1

genre4 = [[0]*10000]
for i in range(4000, 4999):
        norm = np.linalg.norm(train_matrix[i])
        if (norm != 0): 
                instance_vec = train_matrix[i]/norm
                genre4 = genre4 + instance_vec
genres[4] = genre4

n = 5000.0 #size of test set
numAccuratePred = 0
classPrecision = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
classPredictions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
classTestNum = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
for t in range(4849):
        # compute cosine similarity
        scores = {}
        test = test_matrix[t]
        label = Ytest_matrix[t]
        for i in range(5):       
          d = (test*genres[i].transpose()).sum()
          e = np.linalg.norm(test)
          f = np.linalg.norm(genres[i])
          if ((e*f) == 0):
            scores[i] = 0
          else:
            scores[i] = d/(e*f)
        sorted_scores = sorted(scores.iteritems(), key=operator.itemgetter(1),
        reverse=True)
        prediction = sorted_scores[0]
        if (label == prediction[0]):
          numAccuratePred+= 1
          classPrecision[prediction[0]]+= 1
        classPredictions[prediction[0]]+= 1
        classTestNum[label]+= 1
        
accuracy = numAccuratePred/n
print "accuracy " + str(accuracy)
for i in range(5):
  precision = classPrecision[i]/float(classPredictions[i])
  recall = classPrecision[i]/float(classTestNum[i])
  print "for class " + str(i)
  print "precision " + str(precision)
  print "recall " + str(recall)
