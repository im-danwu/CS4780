import os
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import normalize
import subprocess
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
training_file = os.path.join(current_dir, "data/groups2.train")
testing_file = os.path.join(current_dir, "data/groups2.test")
X_train, Y_train = load_svmlight_file(training_file)
X_test, Y_test = load_svmlight_file(testing_file)
norm = normalize(X_train, 'l2')
tnorm = normalize(X_test, 'l2')
dump_svmlight_file(norm, Y_train, 'norm.train', False)
dump_svmlight_file(tnorm, Y_train, 'norm.test', False)
norm_training_file = os.path.join(current_dir, "norm.train")
norm_testing_file = os.path.join(current_dir, "norm.test")
X_norm, Y_train = load_svmlight_file(norm_training_file)
X_tnorm, Y_test = load_svmlight_file(norm_testing_file)

def generate_models(cc = 0.01):
    #for each of the four classes generate a model using svm_learn
    for cl in xrange(4):
        #label a training set marked for each class to be used in svm_learn
        with open('./twocb/c%s_cl%d_norm.train' %(cc, cl+1), 'w') as fp:
            for index in xrange(len(Y_train)):                  #for each training instance
                if Y_train[index] == cl+1:                      #since indexed from 0
                    train_label = '1'
                else:
                    train_label = '-1'
                line = train_label
                train_feat = X_norm[index].indices
                train_feat_value = X_norm[index].data[0]        #same value for every feature of that instance
                for f in train_feat:
                    line = line+" "+str(f+1)+":"+str(train_feat_value)
                fp.write(line+"\n")
        #for each target class (i.e. training set) learn a model
        model = subprocess.call('./svm_light/svm_learn -c %f %s %s' %(0.01, './twocb/c%s_cl%d_norm.train' %(cc, cl+1), './twocb/c%s_m%d_norm.model' %(cc, cl+1)), shell=True)

def training_accuracy(cc = 0.01):
    for cl in xrange(4):
        #label a testing(validation) set for each class to be used in svm_classify
        with open('./twocb/c%s_cl%d_norm.ttrain' %(cc, cl+1), 'w') as fp:
            for index in xrange(len(Y_test)):                   #for each testing instance
                if Y_train[index] == cl+1:                        #since cl indexed from 0
                    train_label = '1'
                else:
                    train_label = '-1'
                line = train_label
                train_feat = X_norm[index].indices
                train_feat_value = X_norm[index].data[0]        #same value for every feature of that instance
                for f in train_feat:
                    line = line+" "+str(f+1)+":"+str(train_feat_value)
                fp.write(line+"\n") #all dem instances
        #for each test set run against a model
        classified = subprocess.call('./svm_light/svm_classify %s %s %s' %('./twocb/c%s_cl%d_norm.ttrain' %(cc, cl+1), './twocb/c%s_m%d_norm.model' %(cc, cl+1), './twocb/c%s_p%d_norm.trainpredict' %(cc, cl+1)), shell = True)

def test_accuracy(cc = 0.01):
    for cl in xrange(4):
        #label a testing(validation) set for each class to be used in svm_classify
        with open('./twocb/c%s_cl%d_norm.test' %(cc, cl+1), 'w') as fp:
            for index in xrange(len(Y_test)):                   #for each testing instance
                if Y_test[index] == cl+1:                        #since cl indexed from 0
                    test_label = '1'
                else:
                    test_label = '-1'
                line = test_label
                test_feat = X_tnorm[index].indices
                test_feat_value = X_tnorm[index].data[0]        #same value for every feature of that instance
                for f in test_feat:
                    line = line+" "+str(f+1)+":"+str(test_feat_value)
                fp.write(line+"\n") #all dem instances
        #for each test set run against a model
        classified = subprocess.call('./svm_light/svm_classify %s %s %s' %('./twocb/c%s_cl%d_norm.test' %(cc, cl+1), './twocb/c%s_m%d_norm.model' %(cc, cl+1), './twocb/c%s_p%d_norm.predict' %(cc, cl+1)), shell = True)

generate_models()
training_accuracy()
test_accuracy()