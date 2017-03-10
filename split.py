from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from util import calulateMetrics
from util import getRandomState
import time

def holdout(data, target, classifier):

    # Prepare a train/test set split
    # split 2/3 1/3 into training & test set
    # We use the random number generator state +1; this will influence how our data is split
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=(getRandomState() + 1))

    start_time_train = time.time()
    classifier.fit(X_train, y_train)
    end_time_train = time.time()

    # predict the test set on our trained classifier
    start_time_test = time.time()
    y_test_predicted = classifier.predict(X_test)
    end_time_test = time.time()

    acc, precision, recall, time_train, time_test = calulateMetrics(y_test, y_test_predicted, start_time_train, end_time_train, start_time_test, end_time_test)
    return [acc, precision, recall, time_train, time_test]

def fold5(data, target, classifier):
    splits = 5
    # Prepare a train/test set split
    kf = KFold(n_splits=splits)
    i = 0
    accSum = []
    precisionSum = []
    recallSum = []
    time_trainSum = []
    time_testSum = []

    for train, test in kf.split(data):
        X_train = data[train]
        X_test = data[test]
        y_train = target[train]
        y_test = target[test]

        # train the k-NN
        start_time_train = time.time()
        classifier.fit(X_train, y_train)
        end_time_train = time.time()

        # predict the test set on our trained classifier
        start_time_test = time.time()
        y_test_predicted = classifier.predict(X_test)
        end_time_test = time.time()

        acc, precision, recall, time_train, time_test = calulateMetrics(y_test, y_test_predicted, start_time_train,
                                                                        end_time_train, start_time_test, end_time_test)
        accSum.append(acc)
        precisionSum.append(precision)
        recallSum.append(recall)
        time_trainSum.append(time_train)
        time_testSum.append(time_test)

    return [accSum, precisionSum, recallSum, time_trainSum, time_testSum]