from sklearn import metrics

def diffToMillisRounded(start, end):
    return round((end - start) * 1000, 3)

def calulateMetrics(y_test, y_test_predicted, start_time_train, end_time_train, start_time_test, end_time_test):
    # Compute metrics
    acc = metrics.accuracy_score(y_test, y_test_predicted)
    precision = metrics.precision_score(y_test, y_test_predicted, average="micro")
    recall = metrics.recall_score(y_test, y_test_predicted, average="micro")
    time_train = diffToMillisRounded(start_time_train, end_time_train)
    time_test = diffToMillisRounded(start_time_test, end_time_test)

    return [acc, precision, recall, time_train, time_test]

def getRandomState():
    # 1510299002 Figl = 29
    # 1510299015 Winterhalder = 33
    # 29+33 = 62
    randomState = 62  # change the state with the numeric parts of your matrikelnummer; if you are in a group, use the sume of the numeric parts
    return randomState