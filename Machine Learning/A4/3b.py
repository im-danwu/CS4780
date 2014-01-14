import os
import subprocess
import numpy as np
import math
from sklearn.datasets import load_svmlight_file
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
testing_file = os.path.join(current_dir, "arxiv/arxiv.test")
training_file = os.path.join(current_dir, "arxiv/arxiv.train")
X_test, Y_test = load_svmlight_file(testing_file)
X_train, Y_train = load_svmlight_file(training_file)#X_train[1] 2nd document Y_train[1] label of 2nd document
#print X_train[1].indices #each word in doc 2
#print X_train[1].data #freq each word in doc 2
num_docs = 29890 
vocab_size = 99757 #number of unique words
neg = .765473402 #Pr(Y = -1)
pos = .234526598 #Pr(Y =  1)
word_count_pos = 0 #number of words in pos class
word_count_neg = 0 #number of words in neg class
word_array_pos = [1]*vocab_size #each word count for positive class
word_array_neg = [1]*vocab_size #each word count for negative class
#number of occurences of w in documents in class y
for i in xrange(num_docs):
    indices = X_train[i].indices
    data = X_train[i].data
    if Y_train[i] > 0:
        for j in xrange(len(indices)):
            #run through the document and add the word count for a given index to word array pos
            word_array_pos[indices[j]-1] = word_array_pos[indices[j]-1] + data[j]
            #update the total number of words in documents with class pos
            word_count_pos = word_count_pos + data[j]
    else:
        for k in xrange(len(indices)):
            word_array_neg[indices[k]-1] = word_array_neg[indices[k]-1] + data[k]
            word_count_neg = word_count_neg + data[k]
print "pos/neg word count"
print word_count_pos
print word_count_neg
print "calculating log values"
#change word array into log of cond probability values 
for i in xrange(vocab_size):
    word_array_pos[i] = math.log(float(word_array_pos[i])/(vocab_size + word_count_pos))
    word_array_neg[i] = math.log(float(word_array_neg[i])/(vocab_size + word_count_neg))
#we need a value pos_sum
#for each test instance, for each word index i, take the ith log value in word_array_pos then
# increment pos_sum by this times the word count of that index
# repeat for this document using word_array_neg
# take the higher of the two values to assign its class
print "calculating test file labels"
test_labels = []
for doc in X_test:
    pos_sum = math.log(pos)
    neg_sum = math.log(neg)
    indx = doc.indices ; word_cnt = doc.data
    for i in xrange(len(indx)):
        pos_sum = pos_sum + word_cnt[i] * word_array_pos[indx[i]-1]
        
        neg_sum = neg_sum + word_cnt[i] * word_array_neg[indx[i]-1]
    if pos_sum + math.log(10) > neg_sum:
        test_labels.append(1)
    else :
        test_labels.append(-1)
print "calculating differences between actual test labels and classfy labels"
error = 0 #number of disagreements between classifying and actual labels
fp = 0 #false pos
fn = 0 #false neg
for lb in xrange(len(Y_test)):
    if Y_test[lb] != test_labels[lb]:
        error = error + 1
        if Y_test[lb] == 1.0:
            fn = fn + 1
        else:
            fp = fp + 1
total_error = float(error)/(len(Y_test))
print "total error is "+str(total_error)
print "false positives: "+str(fp)+'\nfalse negatives: '+str(fn)