import os
import numpy as np
import math
from sklearn.datasets import load_svmlight_file
import scipy
import matplotlib.pyplot as plt
from pdb import set_trace as ss
from random import sample

current_dir = os.path.dirname(os.path.abspath(__file__))
training_file = os.path.join(current_dir, "groups.train")
testing_file = os.path.join(current_dir, "groups.test")

(X_TRAIN, Y_TRAIN) = load_svmlight_file(training_file)
(X_TEST, Y_TEST) = load_svmlight_file(testing_file)

FEATURES = list(xrange(2000))
DOCS = list(xrange(2000))

"""Class for creating the TDIDT. Data holds the feature or the label if a leaf node"""
class Tree(object):
    data, left, right = 0, None, None
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

"""Look for next best feature. If none found i.e. gain = 0 then return.
Returns a Node with a feature and its children or a Node with a label if a leaf"""
def build_tree(S, features, d=0):
    print "CURRENT SIZE OF SET IS "+str(len(S))+" FOR DEPTH "+str(d)
    feat = choose_feature(S, features)
    if not feat or d>=11:
        return Tree(majority_vote(S))
    elif feat:
        passed, failed = split_pf(S, feat)
        features = np.delete(features, [feat]) #OH HAAAYYYYY np.$wag()
        d = d + 1
        return Tree(feat, build_tree(failed, features, d), build_tree(passed, features, d))

def choose_feature(S, features):
    print "choose_feature for size "+str(len(S))
    old_entropy = entropy(S)
    best_gain = 0.0
    best_feat = None

    #compare gains from each feature in features THE KILLER
    for f in features:
        gain = info_gain(split_pf(S, f), old_entropy)
        if gain > best_gain:
            best_gain = gain
            best_feat = f
    print "The best feature found is "+str(best_feat)
    return best_feat
    
"""Takes two sets of doc indices and the pre-split entropy and
returns an info_gain as a float."""
def info_gain(pf_sets, old_entropy):
    new_entropy = 0.0
    passed, failed = pf_sets
    total_size = float(len(passed) + len(failed))
    
    for pf in pf_sets:
        co = len(pf)/total_size #co=|Sv|/|S| for each v
        new_entropy = new_entropy + co*entropy(pf)        
    return old_entropy - new_entropy

"""Takes a set of doc indices and calculates their entropy"""
def entropy(S):
    classes = class_count(S) #(y0, y1, y2, y4)
    entropy = 0.0
    size = float(len(S))
    for c in classes: #c is the size of a class in S
        if(c==len(S)):
            return 0.0
        elif(c!=0):
            entropy = entropy -(c/size)*math.log((c/size), 2)
    return entropy

"""Returns tuple of two arrays of doc indices"""
def split_pf(S, feat):
    passed = []
    failed = []
    for s in S:
        if X_TRAIN[s, feat] > 0.0:
            passed.append(s)
        else:
            failed.append(s)
    return passed, failed    

"""Count the number of each class in S.
Returns a length-4 tuple for count of each class (y0, y1, y2, y4)."""
def class_count(S):
    y0,y1,y2,y4 = 0,0,0,0
    for s in S:
        if Y_TRAIN[s] == 0.0:
            y0 = y0 + 1
        elif Y_TRAIN[s] == 1.0:
            y1 = y1 + 1
        elif Y_TRAIN[s] == 2.0:
            y2 = y2 + 1
        else:
            y4 = y4 + 1
    return (y0,y1,y2,y4)

"""Returns label with highest occurence in S."""
def majority_vote(S):
    majority = float(np.argmax(class_count(S)))
    if majority == 3.0:
        return 4.0
    else:
        return majority

"""Kinda works"""
def print_tree(tree, level=0):
    print " "*level+"Tree "+str(tree.data)
    if not tree.left and not tree.right:
        return
    if tree.left:
        print_tree(tree.left, level+1)
    if tree.right:
        print_tree(tree.right, level+1)

"""Return a label for an instance using a TDIDT."""
def classify(tree, instance):
    if not tree.left and not tree.right:
        return tree.data    
    feature = tree.data
    if instance[0,feature] > 0.0:
        return classify(tree.right, instance) #passed
    else:
        return classify(tree.left, instance) #failed

"""Returns decimal error on a test set given its labels."""
def accuracy(tree, test_set, test_labels):
    labels = []
    mismatch = 0.0    
    for i in test_set:
        label = classify(tree, i)
        labels.append(label)
    for l in range(len(labels)):
        if labels[l] != test_labels[l]:
            mismatch = mismatch + 1
    return mismatch/len(labels)

el_tree = build_tree(DOCS, FEATURES)
score = accuracy(el_tree, X_TEST, Y_TEST)
print score*100