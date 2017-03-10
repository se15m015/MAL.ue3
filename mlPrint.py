import numpy as np

def printKNNHoldout(k, weight, acc, precision, recall, time_train, time_test):
    print()
    print('--------------------------------')
    print('              KNN ' + str(k) + ' weight: ' + weight + ' - Holdout')
    print('--------------------------------')
    print()
    printHoldout(acc, precision, recall, time_train, time_test)
    return

def printKNNFold(k, weight, accSum, precisionSum, recallSum, time_trainSum, time_testSum):
    print()
    print('--------------------------------')
    print('              KNN ' + str(k) + ' weight: ' + weight + ' - Fold 5')
    print('--------------------------------')
    print()
    printFold(accSum, precisionSum, recallSum, time_trainSum, time_testSum)
    return


def printHoldout(acc, precision, recall, time_train, time_test):

    print("accuracy: " + str(acc))
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("time training: " + str(time_train) + " ms ")
    # print("time training: " + str((end_time_train-start_time_train)) + " seconds ")
    print("time test: " + str(time_test) + " ms ")
    # print("test training: " + str((end_time_test - start_time_test)) + " seconds ")
    return

def printFold(accSum, precisionSum, recallSum, time_trainSum, time_testSum):
    print()
    print(" ==== Result ====")
    print("Acc: mean: %s, std: %s" % (np.mean(accSum), np.std(accSum)))
    print("Precision: mean: %s, std: %s" % (np.mean(precisionSum), np.std(precisionSum)))
    print("Recall: mean: %s, std: %s" % (np.mean(recallSum), np.std(recallSum)))
    print("Time Train: mean: %s, std: %s" % (np.mean(time_trainSum), np.std(time_trainSum)))
    print("Time Test: mean: %s, std: %s" % (np.mean(time_testSum), np.std(time_testSum)))
    print()
    return

def printHeader(name, spliter):
    print()
    print('--------------------------------')
    print('              ' + name + ' - ' + spliter)
    print('--------------------------------')
    print()
    return

def printHeaderDataset(name):
    print()
    print('================================')
    print('              ' + name )
    print('================================')
    print()
    return