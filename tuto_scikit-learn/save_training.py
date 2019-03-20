from sklearn import svm
from sklearn import datasets
print("import of svm and datasets done.")

clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()

X,y = iris.data, iris.target # inputs and wanted outputs
clf.fit(X,y)
print("\nThe trained classifier is:\n",clf)

#to save a model thanks to python's built-in persistence model
import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)

N = 2
print("\nDigits predicted:\n",clf2.predict(X[0:N])) # X[0:N] to print predictions of N images
print("\nWanted predictions:\n", y[0:N])

print("\nList of 3 predicted images:\n",list(clf.predict(X[:3])))
clf.fit(X,iris.target_names[y])
print("List of 3 predicted images with associated names:\n",list(clf.predict(X[:3])))

""" In the case of scikit-learn it is more interesting to use joblib
It is more efficient on big data """
print("\nSave learning in 'filename.joblib'\n")
from joblib import dump, load
dump(clf, 'filename.joblib') # to save the trained model

clf = load('filename.joblib') # to load the saved model


