#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

import sys
import re
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from math import *
from collections import defaultdict


# Because of calculation errors, we sometimes end up with negative distances.
# We take here a minimal value of distance, positive (to be able to take the root) and not null (to be able to take the inverse).
MINDIST =  1e-18


class Example:
    """
    An example :
    vector = vector representation of an object (Ovector)
    gold_class = gold class for this object
    """
    def __init__(self, example_number, gold_class):
        self.gold_class = gold_class
        self.example_number = example_number
        self.vector = Ovector()

    def add_feat(self, featname, val):
        self.vector.add_feat(featname, val)


class Ovector:
    """
    Vector representation of an object to classify

    members
    - f= simple dictionnary from feature names to values
         Absent keys correspond to null values
    - norm_square : square value of the norm of this vector
    """
    def __init__(self):
        self.f = defaultdict(lambda: 0.0)   # I modified this line
        self.norm_square = 0 # to be filled later

    def add_feat(self, featname, val=0.0):
        self.f[featname] = val
        self.norm_square = self.norm_square + val * val # norm square is set here


    def prettyprint(self):
        # sort features by decreasing values (-self.f[x])
        #           and by alphabetic order in case of equality
        for feat in sorted(self.f, key=lambda x: (-self.f[x], x)):
            print(feat+"\t"+str(self.f[feat]))

    def distance_to_vector(self, other_vector):
        """ Euclidian distance between self and other_vector,
        Requires: that the .norm_square values be already computed """
        # NB: use the calculation trick
        #   sigma [ (ai - bi)^2 ] = sigma (ai^2) + sigma (bi^2) -2 sigma (ai*bi)
        #                         = norm_square(A) + norm_square(B) - 2 A.B

        return sqrt(self.norm_square + other_vector.norm_square - 2 * self.dot_product(other_vector))

    def dot_product(self, other_vector):
        """ Returns dot product between self and other_vector """
        dot_product = 0
        for featname in self.f:
            dot_product += self.f[featname] * other_vector.f[featname]
        return dot_product

    def cosine(self, other_vector):
        """ Returns cosine of self and other_vector """
        return self.dot_product(other_vector) / (sqrt(self.norm_square) * sqrt(other_vector.norm_square))


class KNN:
    """
    K-NN for document classification (multiclass classification)

    members =

    K = the number of neighbors to consider for taking the majority vote

    examples = list of Example instances

    """
    def __init__(self, examples, K=1, weight_neighbors=None, use_cosine=False, trace=False):
        # examples = list of Example instances
        self.examples = examples
        # the number of neighbors to consider for taking the majority vote
        self.K = K
        # boolean : whether to weight neighbors (by inverse of distance) or not
        self.weight_neighbors = weight_neighbors

        # boolean : whether to use cosine similarity instead of euclidian distance
        self.use_cosine = use_cosine

        # whether to print some traces or not
        self.trace = trace


    def classify(self, ovector):
        """
        K-NN prediction for this ovector,
        for k values from 1 to self.K

        Returns: a K-long list of predicted classes,
        the class at position i is the K-NN prediction when using K=i
        """
        # list of tuples (distance, gold_label) which will be sorted to find the nearest neighbours
        distances_from_ovec = []
        for example in self.examples:
            if self.use_cosine:
                distances_from_ovec.append((-example.vector.cosine(ovector), example.gold_class))   # I added a minus sign because unlike with euclidian distance, the higher the cosinus the more similar the vectors
            else:
                distances_from_ovec.append((example.vector.distance_to_vector(ovector), example.gold_class))
        distances_from_ovec.sort()  # sorts by distance and in case of equality by alphabetical order of the class name (default behaviour in python)
        predictions_by_k = []
        for k in range(1, self.K + 1):
            # dictionnary that associates a class (key) to a number of votes (value)
            majority_vote = defaultdict(lambda: 0)
            for i in range(k):
                if self.weight_neighbors:
                    if self.use_cosine:
                        majority_vote[distances_from_ovec[i][1]] += distances_from_ovec[i][0]   # here we sum because we had stored the opposite of the cosine and in the end the winners are those with a lower value for distance
                    else:
                        majority_vote[distances_from_ovec[i][1]] -= 1/distances_from_ovec[i][0]
                else:
                    majority_vote[distances_from_ovec[i][1]] -= 1   # using negative numbers instead of positive ones helps me sort by votes and then alphabetical order later
            predictions_by_k.append(sorted(majority_vote.items(), key=lambda x:(x[1],x[0]))[0][0])   # [0][0] for the key of the first item of the ordered list
        return predictions_by_k

    def evaluate_on_test_set(self, test_examples):
        """ Runs the K-NN classifier on a list of Example instances
        and evaluates the obtained accuracy

        Returns: a K-long list of accuracies,
        the accuracy at position i is the one obtained when using K=i
        """
        # initialization of the list
        accuracies_by_k = []
        for i in range(self.K):
            accuracies_by_k.append(0.0)
        # counting the numbers of correct predictions for the test_examples
        for example in test_examples:
            predictions_by_k = self.classify(example.vector)
            for i in range(self.K):
                if predictions_by_k[i] == example.gold_class:
                    accuracies_by_k[i] += 1
        # calculating the accuracies:
        accuracies_by_k = [correct_results*100/len(test_examples) for correct_results in accuracies_by_k]
        return accuracies_by_k



def read_examples(infile):
    """ Reads a .examples file and returns a list of Example instances """
    stream = open(infile)
    examples = []
    example = None
    while 1:
        line = stream.readline()
        if not line:
            break
        line = line[0:-1]
        if line.startswith("EXAMPLE_NB"):
            if example != None:
                examples.append(example)
            cols = line.split('\t')
            gold_class = cols[3]
            example_number = cols[1]
            example = Example(example_number, gold_class)
        elif line and example != None:
            (featname, val) = line.split('\t')
            example.add_feat(featname, float(val))


    if example != None:
        examples.append(example)
    return examples



usage = """ K-NN DOCUMENT CLASSIFIER

  """+sys.argv[0]+""" [options] TRAIN_FILE TEST_FILE

  TRAIN_FILE and TEST_FILE are in *.examples format

"""

parser = argparse.ArgumentParser(usage = usage)
parser.add_argument('train_file', default=None, help='Examples that will be used for the K-NN prediction (in .examples format)')
parser.add_argument('test_file', default=None, help='Test examples de test (in .examples format)')
parser.add_argument('-k', "--k", default=1, type=int, help='Hyperparameter K : maximum number of neighbors to consider (all values between 1 and k will be tested). Default=1')
parser.add_argument('-v', "--trace", action="store_true", help="Toggles the verbose mode. Default=False")
parser.add_argument('-w', "--weight_neighbors", action="store_true", help="If set, neighbors will be weighted before majority vote. If cosine: cosine weighting, if distance, weighting using the inverse of the distance. Default=False")
parser.add_argument('-c', "--use_cosine", action="store_true", help="Toggles the use of cosine similarity instead of euclidian distance. Default=False")
parser.add_argument('-p', "--hyperparameters_test", action="store_true", help="Toggles the test and subsequent plotting of the four hyperparameters combinations (besides k). Default=False")
args = parser.parse_args()

#------------------------------------------------------------
# Loading training examples
training_examples = read_examples(args.train_file)
# Loading test examples
test_examples = read_examples(args.test_file)

myclassifier = KNN(examples = training_examples,
                   K = args.k,
                   weight_neighbors = args.weight_neighbors,
                   use_cosine = args.use_cosine,
                   trace=args.trace)

# classification and evaluation on test examples
accuracies = myclassifier.evaluate_on_test_set(test_examples)

for i in range(args.k):
    print("ACCURRACY FOR k = " + str(i+1) + " = " + str(accuracies[i]))

# plotting results for all four combinations of hyperparameters
if args.hyperparameters_test:

    classifier_distance = KNN(examples = training_examples,
                       K = args.k,
                       weight_neighbors = False,
                       use_cosine = False,
                       trace=args.trace)
    accuracies_distance = classifier_distance.evaluate_on_test_set(test_examples)
    classifier_distance_weight = KNN(examples = training_examples,
                       K = args.k,
                       weight_neighbors = True,
                       use_cosine = False,
                       trace=args.trace)
    accuracies_distance_weight = classifier_distance_weight.evaluate_on_test_set(test_examples)
    classifier_cosine = KNN(examples = training_examples,
                       K = args.k,
                       weight_neighbors = False,
                       use_cosine = True,
                       trace=args.trace)
    accuracies_cosine = classifier_cosine.evaluate_on_test_set(test_examples)
    classifier_cosine_weight = KNN(examples = training_examples,
                       K = args.k,
                       weight_neighbors = True,
                       use_cosine = True,
                       trace=args.trace)
    accuracies_cosine_weight = classifier_cosine_weight.evaluate_on_test_set(test_examples)
    list_ks = list(range(1, args.k+1))

    df = pd.DataFrame({"Value of k": list_ks, "accuracies using euclidian distances and not weighting the neighbors": accuracies_distance, "accuracies using euclidian distances and weighting the neighbors": accuracies_distance_weight, "accuracies using cosine and not weighting the neighbors": accuracies_cosine, "accuracies using cosine and weighting the neighbors": accuracies_cosine_weight})
    df.plot(title = "Classifier accuracies for different hyperparameter settings", x = "Value of k", y = {"accuracies using euclidian distances and not weighting the neighbors", "accuracies using euclidian distances and weighting the neighbors", "accuracies using cosine and not weighting the neighbors", "accuracies using cosine and weighting the neighbors"}, ylabel = "Accuracy of the classifier", style = '.-')
    plt.show()
