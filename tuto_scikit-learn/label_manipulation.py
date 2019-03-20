from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

X = [
    [1,2],
    [2,4],
    [4,5],
    [3,2],
    [3,1],
]

y = [0, 0, 1, 1, 2]

print("input arrays:\n",X,"\nWanted output for each input array:\n",y)

""" Using multiclass classifiers as OneVsRestClassifier,
the learning and prediction tasks are dependent on the format of the target data """
classif = OneVsRestClassifier(estimator=SVC(gamma='scale', random_state=0))

print("\nTraining and result of prediction of input data:\n",classif.fit(X,y).predict(X))

""" Now we can change the label shape
Above, it has 3 possible values: 0,1 or 2
Below we change the shape to 
0: 1 0 0
1: 0 1 0
2: 0 0 1 """

y = LabelBinarizer().fit_transform(y)
print("Label data binarized:\n",y)

print("\nTraining and result of prediction of input data binarized:\n",classif.fit(X,y).predict(X))

""" We can see that the prediction changes
with the label transformation, without the transformation
we obtained good predictions contrarily to binarized labels...
When we have 0 0 0 it significates that none of the labels is recognized """

y2 = [
    [0,1],
    [0,2],
    [1,3],
    [0,2,3],
    [2,4],
]
""" Here we have 5 possible outputs (0 to 4)
 """
print("\nNew set of label:\n",y2)
y2 = MultiLabelBinarizer().fit_transform(y2)
print("Multi Label data binarized:\n",y2)

""" Let's try to train a neural network that
predict correctly these labels"""

clf = OneVsRestClassifier(estimator=SVC(kernel='rbf', gamma='scale') )
clf.fit(X,y2) #train
print("\nOutput predictions:\n", list(clf.predict(X[0:1])))
print("Output wanted:\n", list(y2[0:1]))
