# Import statements
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn import neighbors
from sklearn import metrics
from sklearn.utils import shuffle
import time
import datetime

# load the Iris dataset
dataSet = datasets.load_iris()

# Shuffle our input data
#1510299002 Figl = 29
#1510299015 Winterhalder = 33
#29+33 = 62
randomState = 62 # change the state with the numeric parts of your matrikelnummer; if you are in a group, use the sume of the numeric parts

data, target = shuffle(dataSet.data, dataSet.target, random_state=randomState)

# Prepare a train/test set split
#kf = train_test_split(n_splits=2)


# split 2/3 1/3 into training & test set
# We use the random number generator state +1; this will influence how our data is split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=(randomState+1))

print("Data-orig")
print(len(dataSet.data))
for do in dataSet.data:
    print(do)

print("Target-orig")
print(len(dataSet.target))
for to in dataSet.target:
    print(to)

print("Data")
print(len(data))
for d in data:
    print(d)

print("target")
print(len(target))
for t in target:
    print(t)

print("X_train")
print(len(X_train))
for xtr in X_train:
    print(xtr)

print("X_test")
print(len(X_test))
for xte in X_test:
    print(xte)

print("y_train")
print(len(y_train))
for ytr in y_train:
    print(ytr)

print("y_test")
print(len(y_test))
for yte in y_test:
    print(yte)



# Prepare a train/test set split
kf = KFold(n_splits=5)
for train, test in kf.split(data):
    print("for")
    print("Train: %s" % ( train ))
    print("Test: %s" % ( test ))