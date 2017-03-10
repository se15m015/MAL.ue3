# Import statements
import numpy as np

from sklearn.model_selection import KFold
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

splits = 5
# Prepare a train/test set split
kf = KFold(n_splits=splits)
i = 0
accSum = []
precisionSum = []
recallSum = []
time_trainSum = []
time_testSum = []

# parameters for k-NN
n_neighbors = [1, 5, 13, 55]
weights = ["uniform", "distance"]

for k in n_neighbors:
    for weight in weights:
        print()
        print('================================================')
        print('              KNN ' + str(k) + ' weight: ' + weight)
        print('================================================')
        print()
        for train, test in kf.split(dataSet.data):

            print()
            print(" ----" + str(i+1) + " Split ----")

            X_train = data[train]
            X_test = data[test]
            y_train = target[train]
            y_test = target[test]

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
            accSum.append(acc)
            precision = metrics.precision_score(y_test, y_test_predicted, average="micro")
            precisionSum.append(precision)
            recall = metrics.recall_score(y_test, y_test_predicted, average="micro")
            recallSum.append(precision)

            time_train = diffToMillisRounded(start_time_train, end_time_train)
            time_trainSum.append(time_train)

            time_test = diffToMillisRounded(start_time_test, end_time_test)
            time_testSum.append(time_test)

            print()
            print("accuracy: " + str(acc))
            print("precision: " + str(precision))
            print("recall: " + str(recall))
            print("time training: " + str(time_train) + " ms ")
            # print("time training: " + str((end_time_train-start_time_train)) + " seconds ")
            print("time test: " + str(time_test) + " ms ")
            # print("test training: " + str((end_time_test - start_time_test)) + " seconds ")
            i += 1

        print()
        print(" ==== Result ====")
        print("Acc: mean: %s, std: %s" % (np.mean(accSum), np.std(accSum)))
        print("Precision: mean: %s, std: %s" % (np.mean(precisionSum), np.std(precisionSum)))
        print("Recall: mean: %s, std: %s" % (np.mean(recallSum), np.std(recallSum)))
        print("Time Train: mean: %s, std: %s" % (np.mean(time_trainSum), np.std(time_trainSum)))
        print("Time Test: mean: %s, std: %s" % (np.mean(time_testSum), np.std(time_testSum)))
        print()

        accSum = []
        precisionSum = []
        recallSum = []
        i = 0