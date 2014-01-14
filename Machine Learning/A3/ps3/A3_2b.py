import os
from sklearn.datasets import load_svmlight_file
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
training_file = os.path.join(current_dir, "data/groups2.train")
testing_file = os.path.join(current_dir, "data/groups2.test")
X_train, Y_train = load_svmlight_file(training_file)
X_test, Y_test = load_svmlight_file(testing_file)

def generate_models(cc = 0.01):
    #for each of the four classes generate a model using svm_learn
    for cl in xrange(4):
        #label a training set marked for each class to be used in svm_learn
        with open('./twob/c%s_cl%d.train' %(cc, cl+1), 'w') as fp:
            for index in xrange(len(Y_train)):                  #for each training instance
                if Y_train[index] == cl+1:                      #since indexed from 0
                    train_label = '1'
                else:
                    train_label = '-1'
                line = train_label
                train_feat = X_train[index].indices
                for f in train_feat:
                    line = line+" "+str(f+1)+":1"
                fp.write(line+"\n")
        #for each target class (i.e. training set) learn a model
        model = subprocess.call('./svm_light/svm_learn -c %f %s %s' %(cc, './twob/c%s_cl%d.train' %(cc, cl+1), './twob/c%s_m%d.model' %(cc, cl+1)), shell=True)

def training_accuracy(cc = 0.01):
    for cl in xrange(4):
        #label a testing(validation) set for each class to be used in svm_classify
        with open('./twob/c%s_t%d.train' %(cc, cl+1), 'w') as fp:
            for index in xrange(len(Y_test)):                   #for each testing instance
                if Y_train[index] == cl+1:                        #since cl indexed from 0
                    train_label = '1'
                else:
                    train_label = '-1'
                line = train_label
                train_feat = X_train[index].indices
                for f in train_feat:
                    line = line+" "+str(f+1)+":1"
                fp.write(line+"\n") #all dem instances
        #for each test set run against a model
        classified = subprocess.call('./svm_light/svm_classify %s %s %s' %('./twob/c%s_t%d.train' %(cc, cl+1), './twob/c%s_m%d.model' %(cc, cl+1), './twob/c%s_p%d.trainpredict' %(cc, cl+1)), shell = True)

def test_accuracy(cc = 0.01):
    for cl in xrange(4):
        #label a testing(validation) set for each class to be used in svm_classify
        with open('./twob/c%s_t%d.test' %(cc, cl+1), 'w') as fp:
            for index in xrange(len(Y_test)):                   #for each testing instance
                if Y_test[index] == cl+1:                        #since cl indexed from 0
                    test_label = '1'
                else:
                    test_label = '-1'
                line = test_label
                test_feat = X_test[index].indices
                for f in test_feat:
                    line = line+" "+str(f+1)+":1"
                fp.write(line+"\n") #all dem instances
        #for each test set run against a model
        classified = subprocess.call('./svm_light/svm_classify %s %s %s' %('./twob/c%s_t%d.test' %(cc, cl+1), './twob/c%s_m%d.model' %(cc, cl+1), './twob/c%s_p%d.predict' %(cc, cl+1)), shell = True)

generate_models()
training_accuracy()
test_accuracy()