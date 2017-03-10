# Import statements
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from util import getRandomState
from classifier import kNN
from classifier import decisionTree
from classifier import naiveBayes
from classifier import perceptron
from mlPrint import printHeaderDataset
import csv

# load the IRIS dataset
printHeaderDataset("IRIS")
dataSet = datasets.load_iris()
# Shuffle our input data
data, target = shuffle(dataSet.data, dataSet.target, random_state=getRandomState())

kNN(data, target)
decisionTree(data, target)
naiveBayes(data, target)
perceptron(data, target)

# load the DIGITS dataset
printHeaderDataset("DIGITS")
dataSet = datasets.load_digits()
# Shuffle our input data
data, target = shuffle(dataSet.data, dataSet.target, random_state=getRandomState())

kNN(data, target)
decisionTree(data, target)
naiveBayes(data, target)
perceptron(data, target)

# load the BEAST-CANCER dataset
printHeaderDataset("BREAST-CANCER")
dataSet = datasets.load_breast_cancer()

# Shuffle our input data
data, target = shuffle(dataSet.data, dataSet.target, random_state=getRandomState())

kNN(data, target)
decisionTree(data, target)
naiveBayes(data, target)
perceptron(data, target)


# load the BEAST-CANCER-FROM-FILE dataset
printHeaderDataset("BREAST-CANCER-FROM-FILE")
dataSet = datasets.load_breast_cancer()

with open('data/breast-cancer-wisconsin.data') as csv_file:
    data_file = csv.reader(csv_file)
    data = np.empty((683, 9))
    target = np.empty((683,), dtype=np.int)

    i = 0

    for count, value in enumerate(data_file):
        #remove id using 1:
        # print(value[1:-1])
        # print(value[-1])
        # print(count)
        # print(i)
        if not any('?' in s for s in value):
            data[i] = np.asarray(value[1:-1], dtype=np.int)
            target[i] = np.asarray(value[-1], dtype=np.int)
            i += 1

# # Shuffle our input data
data, target = shuffle(dataSet.data, dataSet.target, random_state=getRandomState())

kNN(data, target)
decisionTree(data, target)
naiveBayes(data, target)
perceptron(data, target)