# Import statements
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import neighbors
from sklearn import metrics
from sklearn.utils import shuffle
import time
import datetime
from util import diffToMillisRounded

# load the Iris dataset
dataSet = datasets.load_iris()


# Shuffle our input data
#1510299002 Figl = 29
#1510299015 Winterhalder = 33
#29+33 = 62
randomState = 62 # change the state with the numeric parts of your matrikelnummer; if you are in a group, use the sume of the numeric parts
data, target = shuffle(dataSet.data, dataSet.target, random_state=randomState)

# Prepare a train/test set split
# split 2/3 1/3 into training & test set
# We use the random number generator state +1; this will influence how our data is split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=(randomState+1))

# parameters for k-NN
n_neighbors = [1, 5, 13, 55]
weights = ["uniform", "distance"]

for k in n_neighbors:
    for weight in weights:
        # train the k-NN
        classifier = neighbors.KNeighborsClassifier(k, weights=weight)
        start_time_train = time.time()
        classifier.fit(X_train, y_train)
        end_time_train = time.time()

        # predict the test set on our trained classifier
        start_time_test = time.time()
        y_test_predicted = classifier.predict(X_test)
        end_time_test = time.time()

        # Compute metrics
        acc = metrics.accuracy_score(y_test, y_test_predicted)
        precision = metrics.precision_score(y_test, y_test_predicted, average="micro")
        recall = metrics.recall_score(y_test, y_test_predicted, average="micro")
        time_train = diffToMillisRounded(start_time_train, end_time_train)
        time_test = diffToMillisRounded(start_time_test, end_time_test)

        print()
        print('--------------------------------')
        print('              KNN ' + str(k) + ' weight: ' + weight)
        print('--------------------------------')
        print()
        print("accuracy: " + str(acc))
        print("precision: " + str(precision))
        print("recall: " + str(recall))
        print("time training: " + str(time_train) + " ms ")
        # print("time training: " + str((end_time_train-start_time_train)) + " seconds ")
        print("time test: " + str(time_test) + " ms ")
        # print("test training: " + str((end_time_test - start_time_test)) + " seconds ")