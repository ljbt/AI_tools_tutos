import numpy as np
from sklearn import random_projection
from sklearn.svm import SVC

rng = np.random.RandomState(0)
X = rng.rand(10, 2000) # initialise 10 arrays with 2000 values 
#print(X)

X = np.array(X, dtype='float32')
print("\nArray including random values of type",X.dtype,":\n",X)


y = rng.binomial(1 , 0.5 , 10) # creates array with random values: 1 or 0
print("\nOutput wanted for each input array:\n",y)


"""  Be carefull with the data shape 
We have here 10 arrays of 2000 values, so we have 10 outputs
one for each input array """
print("\nTest changing the neural network parameters:\n")
clf = SVC()
clf.set_params(kernel='linear').fit(X,y) # X_new or X, as we want
print("Neural network parameters:\n", clf)
print("Predictions of first 3 array inputs:",clf.predict(X[0:3]),"\nWanted outputs:",y[0:3])

clf.set_params(kernel='rbf', gamma='scale').fit(X,y) # X_new or X, as we want
print("\nNeural network parameters:\n", clf)
print("Predictions of first 3 array inputs:",clf.predict(X[0:3]),"\nWanted outputs:",y[0:3])

