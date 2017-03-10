from sklearn import neighbors
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from split import holdout
from split import fold5
from mlPrint import printKNNHoldout
from mlPrint import printKNNFold
from mlPrint import printHoldout
from mlPrint import printFold
from mlPrint import printHeader
from sklearn.neural_network import MLPClassifier
from util import getRandomState

def kNN(data, target):
    # parameters for k-NN
    n_neighbors = [1, 5, 13, 55]
    weights = ["uniform", "distance"]

    for k in n_neighbors:
        for weight in weights:
            # train the k-NN
            classifier = neighbors.KNeighborsClassifier(k, weights=weight)

            #Holdout
            acc, precision, recall, time_train, time_test = holdout(data, target, classifier)
            printKNNHoldout(k, weight, acc, precision, recall, time_train, time_test)

            #Fold 5
            accSum, precisionSum, recallSum, time_trainSum, time_testSum = fold5(data, target, classifier)
            printKNNFold(k, weight, accSum, precisionSum, recallSum, time_trainSum, time_testSum)
    return

def decisionTree(data, target):
    #Holdout
    classifier = tree.DecisionTreeClassifier()
    acc, precision, recall, time_train, time_test = holdout(data, target, classifier)
    printHeader("Decision Tree", "Holdout")
    printHoldout(acc, precision, recall, time_train, time_test)

    #Fold 5
    acc, precision, recall, time_train, time_test = fold5(data, target, classifier)
    printHeader("Decision Tree", "Fold 5")
    printFold(acc, precision, recall, time_train, time_test)
    return

def naiveBayes(data, target):
    classifier = GaussianNB()

    #Holdout
    acc, precision, recall, time_train, time_test = holdout(data, target, classifier)
    printHeader("Naive Bayes", "Holdout")
    printHoldout(acc, precision, recall, time_train, time_test)

    # Fold 5
    acc, precision, recall, time_train, time_test = fold5(data, target, classifier)
    printHeader("Naive Bayes", "Fold 5")
    printFold(acc, precision, recall, time_train, time_test)
    return

def perceptron(data, target):
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = getRandomState())

    # Holdout
    acc, precision, recall, time_train, time_test = holdout(data, target, classifier)
    printHeader("Perceptron", "Holdout")
    printHoldout(acc, precision, recall, time_train, time_test)

    # Fold 5
    acc, precision, recall, time_train, time_test = fold5(data, target, classifier)
    printHeader("Perceptron", "Fold 5")
    printFold(acc, precision, recall, time_train, time_test)
    return
