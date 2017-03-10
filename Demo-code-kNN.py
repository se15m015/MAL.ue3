print('--------------------------------')
print('           KNN Demo')
print('--------------------------------')
print()

# Import statements
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import neighbors
from sklearn import metrics
from sklearn.utils import shuffle
import time
import datetime

# load the Iris dataset
dataSet = datasets.load_iris()


# Shuffle our input data
randomState=42 # change the state with the numeric parts of your matrikelnummer; if you are in a group, use the sume of the numeric parts
data, target = shuffle(dataSet.data, dataSet.target, random_state=randomState)

# Prepare a train/test set split
# split 2/3 1/3 into training & test set
# We use the random number generator state +1; this will influence how our data is split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=(randomState+1))

# parameters for k-NN
n_neighbors = 15

# train the k-NN
classifier = neighbors.KNeighborsClassifier(n_neighbors)
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

print("accuracy: " + str(acc))
print("precision: " + str(precision))
print("time training: " + str(end_time_train-start_time_train) + " seconds")
print("time training: " + str(end_time_test-start_time_test) + " seconds")