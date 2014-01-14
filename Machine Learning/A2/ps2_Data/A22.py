import os
import numpy as np
import math
from sklearn.datasets import load_svmlight_file
import scipy
import matplotlib.pyplot as plt
from pdb import set_trace as ss
from random import sample

current_dir = os.path.dirname(os.path.abspath(__file__))
training_file = os.path.join(current_dir, "circle.train")

X_TRAIN, Y_TRAIN = load_svmlight_file(training_file)
X_TRAIN = X_TRAIN.toarray().tolist()
Y_TRAIN = Y_TRAIN.tolist()

#Array of all instances as tuples of form S((x1, Y1), (x2, Y2), ..., (xn, Yn))
#S[0] instance, S[0][0] coordinates, S[0][0][0] x-coord, S[0][0][1] y-coord, S[0][0][1] label
ALL_INSTANCES = []
for i in range(len(X_TRAIN)):
    ALL_INSTANCES.append([X_TRAIN[i], Y_TRAIN[i]])

"""Error for when no more thresholds give any info gain. Tree should stop growing."""
class NoThresholdError(Exception):
    def __init__(self):
        self.message = "NO MORE THRESHOLDS"
    def __str__(self):
        return repr(self.message)

"""Class for creating the TDIDT. Data holds the splitting criterion ("attr", label) or the label if a leaf node"""
class Node(object):
    data, left, right = 0, None, None
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

"""Checks for base cases of empty set or set of instances of one class only.
Else look for next best threshold to split on. If none found, then break tie and end.
Returns a Node with a threshold and its children or a Node with a label if a leaf"""
def build_tree(S):
    if len(S) == 0: #empty S, negative is default
        return Node(0.0)
    pos, neg = split_set_on_class(S) #check not all s in S have same class or one is empty
    if neg == 0:
        return Node(1.0)
    elif pos == 0:
        return Node(0.0)

    else:
        try: #if a better threshold is available
            threshold = choose_threshold(S) 
            passed, failed = split_set_on_threshold(S, threshold[0], threshold[1])
            return Node(threshold, build_tree(failed), build_tree(passed)) #continue recursing after splitting on threshold
        except NoThresholdError as e: #no thresholds yield info gain -> reached max depth of this branch
            return Node(majority(S))

"""Takes in a training set. Finds best attribute-threshold pair not used yet
by comparing info gain from each possible split.
Returns a tuple of form ("attr", n) to use for next node or None if no threshold gives positive gain."""
def choose_threshold(S):
    pos, neg = split_set_on_class(S)
    curr_entropy = entropy(pos, neg) #used to calculate gain
    best_gain = 0.0 #might be better to store gains from each threshold in numpy array and then use numpy select max function
    best_pair = None

    #checking thresholds using x
    for i in range(10):
        passed, failed = split_set_on_threshold(S, "x", i)
        gain = info_gain([passed, failed], curr_entropy)
        if gain > best_gain:
            best_gain = gain
            best_pair = ("x", i)

    # and then y
    for i in range(10):
        passed, failed = split_set_on_threshold(S, "y", i)
        gain = info_gain([passed, failed], curr_entropy)
        if gain > best_gain:
            best_gain = gain
            best_pair = ("y", i)
    
    if not best_pair:
        raise NoThresholdError()
    return best_pair

"""Splits the set into two groups based on attribute-threshold pair.
Returns array of two arrays of instances"""
def split_set_on_threshold(S, attribute, threshold):
    passed = []
    failed = []
    if attribute == "x":
        for instance in S:
            if instance[0][0] >= threshold:
                passed.append(instance)
            else:
                failed.append(instance)
    elif attribute == "y":
        for instance in S:
            if instance[0][1] >= threshold:
                passed.append(instance)
            else:
                failed.append(instance)
    return [passed, failed]

"""Sort data into different sets based on their labels. S[i][1] gives the label for instance i in S.
Returns size of each class as an array of integers"""
def split_set_on_class(S):
    positives = 0
    negatives = 0
    for instance in S:
        label = instance[1]
        if label == 1.0:
            positives = positives + 1
        else:
            negatives = negatives + 1
    return [positives, negatives]

"""Calculate the net information gain from old entropy from entropies of array of sets.
Returns the gain as a float."""
def info_gain(sets, old_entropy): #sets contains passed, failed
    new_entropy = 0.0
    total_size = 0.0 #used to calculate entropy of the subsets from after split
    for s in sets:
        total_size = total_size + len(s)

    for s in sets:
        if total_size > 0: #calculate entropy only if set is not empty
            pos, neg = split_set_on_class(s)
            co = len(s)/total_size #coefficient |Sv|/|S|
            new_entropy = new_entropy + co*entropy(pos, neg)
    return old_entropy - new_entropy

"""Calculate the entropy of a set.
Returns a float 0.0 to 1.0, 0.0 if no entropy and 1.0 if even split"""
def entropy(n1, n2):
    #entropy is 0 if either or both sets empty (data is pure)
    if n1 == 0 or n2 == 0:
        return 0.0
    
    #calculate entropy
    else:
        total = float(n1+n2)
        s1 = n1/total
        s2 = n2/total
        entropy = -s1*math.log(s1, 2) -s2*math.log(s2, 2)
        return entropy

"""Return label for a set using majority vote. Go with neg if tied"""
def majority(S):
    pos, neg = split_set_on_class(S)
    if pos > neg:
        return 1.0
    else:
        return 0.0

"""Return a label for an instance using a TDIDT."""
def run_through_tree(tdidt, instance):
    if not tdidt.left and not tdidt.right:
        return tdidt.data

    threshold = tdidt.data #(attr, n)
    if threshold[0] == "x":
        if instance[0][0] >= threshold[1]:
            return run_through_tree(tdidt.right, instance) #passed
        else:
            return run_through_tree(tdidt.left, instance) #failed
    if threshold[0] == "y":
        if instance[0][1] >= threshold[1]:
            return run_through_tree(tdidt.right, instance) #passed
        else:
            return run_through_tree(tdidt.left, instance) #failed

def plot(tdidt):
    pos_x, pos_y = [], []
    neg_x, neg_y = [], []
    x, y = 0.0, 0.0
    while x <= 10:
        while y <= 10:
            instance = [[x,y], 0.0]
            label = run_through_tree(tdidt, instance)
            if label == 1.0:
                pos_x.append(instance[0][0])
                pos_y.append(instance[0][1])
            else:
                neg_x.append(instance[0][0])
                neg_y.append(instance[0][1])
            y = y + 0.10
        y = 0.0
        x = x + 0.10
    plt.plot(pos_x, pos_y, 'ro')
    plt.plot(neg_x, neg_y, 'go')
    plt.axis([0,10,0,10])
    plt.show()
#    plt.savefig('2b_tree')

"""Kinda works"""
def print_tree(tree, level=0):
    print "\t"*level+"Node "+str(tree.data)
    if not tree.left and not tree.right:
        return
    if tree.left:
        print_tree(tree.left, level+1)
    if tree.right:
        print_tree(tree.right, level+1)

def tree_average(tree_list, inst):
    vote = 0
    for t in tree_list:
        label = run_through_tree(t, inst)
        if label == 0:
            vote = vote - 1
        else:
            vote = vote + 1
    if vote > 0:
        return 1.0
    else:
        return 0.0
    
def combined_plot(M, S, p):
    pos_x, pos_y = [], []
    neg_x, neg_y = [], []
    x, y = 0.0, 0.0
    tree_list = build_M_trees(M, S, p)
    while x <= 10:
        while y <= 10:
            instance = [[x,y], 0.0]
            label = tree_average(tree_list, instance)
            if label == 1.0:
                pos_x.append(instance[0][0])
                pos_y.append(instance[0][1])
            else:
                neg_x.append(instance[0][0])
                neg_y.append(instance[0][1])
            y = y + 0.10
        y = 0.0
        x = x + 0.10
    plt.plot(pos_x, pos_y, 'ro')
    plt.plot(neg_x, neg_y, 'go')
    plt.axis([0,10,0,10])
    plt.show()

"""Trains M number of trees by random sampling without replacement on set S"""
def build_M_trees(M, S, p): #M is number of trees, S is a training set, p=% of S for each subset
    all_trees = []
    for i in range(M):
        Si = sample_set(S, p) #sample_set takes a random 20% sample from S
        tree = build_tree(Si)
        #plot(tree) #commented out to save time
        all_trees.append(tree)
    return all_trees

""""Returns a sample of a set based on percentage (float)"""
def sample_set(S, p):
    sample_size = int(len(S)*p)
    return sample(S, sample_size)

combined_plot(101, ALL_INSTANCES, .2)