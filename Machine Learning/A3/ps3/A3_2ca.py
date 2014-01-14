import os
import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import sys
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
training_file = os.path.join(current_dir, "data/groups2.train")
testing_file = os.path.join(current_dir, "data/groups2.test")
X_train, Y_train = load_svmlight_file(training_file)
X_test, Y_test = load_svmlight_file(testing_file)
norm = normalize(X_train)
tnorm = normalize(X_test)
dump_svmlight_file(norm, Y_train, 'norm.train', False)
dump_svmlight_file(tnorm, Y_train, 'norm.test', False)
norm_training_file = os.path.join(current_dir, "norm.train")
X_norm, Y = load_svmlight_file(norm_training_file)

accuracies = {0.001:None, 0.01:None, 0.1:None, 0.5:None, 1.0:None, 2.0:None, 5.0:None, 10.0:None, 100.0:None}
C = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]

kf = cross_validation.KFold(2000, 5, shuffle=True)
for cc in C:
    #round robin for len(kf) times to get len(kf) accuracies
    sum_of_accuracies = 0
    rr = 1
    for train_index, test_index in kf:
        X_train, X_test = X_norm[train_index], X_norm[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        #for each of the four classes generate a model using svm_learn
        for cl in xrange(4):
            #write a file to label a training set marked for each class to be used in svm_learn
            with open('./twoca/c%s_r%d_cl%d.train' %(cc, rr, cl+1), 'w') as fp:
                for index in xrange(len(Y_train)):                  #for each training instance
                    if Y_train[index] == cl+1:                      #since indexed from 0
                        train_label = '1'
                    else:
                        train_label = '-1'
                    line = train_label
                    train_feat = np.flipud(X_train[index].indices)
                    train_feat_value = X_train[index].data[0]
                    for f in train_feat:
                        line = line+" "+str(f+1)+":"+str(train_feat_value)
                    fp.write(line+"\n")
            #for each target class (i.e. training set) learn a model
            model = subprocess.call('./svm_light/svm_learn -c %f %s %s' %(cc, './twoca/c%s_r%d_cl%d.train' %(cc, rr, cl+1), './twoca/c%s_r%d_m%d.model' %(cc, rr, cl+1)), shell=True)

        #for each model, each corresponding to one class, generate a prediction
        for cl in xrange(4):
            #write a file to label a testing(validation) set for each class to be used in svm_classify
            with open('./twoca/c%s_r%d_t%d.test' %(cc, rr, cl+1), 'w') as fp:
                for index in xrange(len(Y_test)):                   #for each testing instance
                    if Y_test[index] == cl+1:                        #since cl indexed from 0
                        test_label = '1'
                    else:
                        test_label = '-1'
                    line = test_label
                    test_feat = np.flipud(X_test[index].indices)
                    train_feat_value = X_train[index].data[0]
                    for f in train_feat:
                        line = line+" "+str(f+1)+":"+str(train_feat_value)
                    fp.write(line+"\n") #all dem instances
            #for each test set run against a model
            classified = subprocess.call('./svm_light/svm_classify %s %s %s' %('./twoca/c%s_r%d_t%d.test' %(cc, rr, cl+1), './twoca/c%s_r%d_m%d.model' %(cc, rr, cl+1), './twoca/c%s_r%d_p%d.predict' %(cc, rr, cl+1)), shell = True)
        
        #for each instance in the testing set but now w/o labels, label that instance with the prediction with the highest (absolute?) margin
        maximums = list(range(400))
        #open all four files and read each label for the 400 instances
        predictions_1 = None
        predictions_2 = None
        predictions_3 = None
        predictions_4 = None
        
        with open('./twoca/c%s_r%d_p%d.predict' %(cc, rr, 1)) as fp:
            predictions_1 = [float(line.rstrip('\n')) for line in fp]
            
        with open('./twoca/c%s_r%d_p%d.predict' %(cc, rr, 2)) as fp:
            predictions_2 = [float(line.rstrip('\n')) for line in fp]
        
        with open('./twoca/c%s_r%d_p%d.predict' %(cc, rr, 3)) as fp:
            predictions_3 = [float(line.rstrip('\n')) for line in fp]
        
        with open('./twoca/c%s_r%d_p%d.predict' %(cc, rr, 4)) as fp:
            predictions_4 = [float(line.rstrip('\n')) for line in fp]
        
        for i in range(400):
            compare = ( predictions_1[i], predictions_2[i], predictions_3[i], predictions_4[i] )
            maximums[i] = compare.index(max(compare))+1
        #calculate error for this round and store
        #compare maximums to Y_test for accuracy
        correct = 0.0
        for l in range(len(Y_test)):
            if Y_test[l] == maximums[l]:
                correct += 1.0
        sum_of_accuracies = sum_of_accuracies + (correct/len(Y_test))*100
        rr += 1
    accuracies[cc] = sum_of_accuracies/len(kf)
with open('./twoca/accuracies.txt', 'w') as fp:
    fp.write(str(accuracies))