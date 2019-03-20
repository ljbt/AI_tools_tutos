from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
digits = datasets.load_digits()

print("import of the datasets done")

print("\ndigits data (all images): \n",digits.data) # print images arrays
print("\nwanted outputs:\n",digits.target) # the wanted values for each image

print("\nfirst image:\n",digits.images[0]) # print first image


"""  example of estimator
 estimators can implement the methods fit(X,y) and predict(T)
here we put 0.001 as gamma, we can use grid search and cross validation to find good values
 """
classifier = svm.SVC(gamma=0.001, C=100.)  
print("\nClassifier initialized")

""" Now we have to learn from the model
For that we use the fit method, and gives all the data, 
except the final image that we will use for our predicting """
classifier.fit(digits.data[:-1], digits.target[:-1])
print("\nClassifier trained")
print(classifier)

print("\nThe last image is predicted as a",classifier.predict(digits.data[-1:]))
