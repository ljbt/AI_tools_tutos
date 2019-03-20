import numpy as np
import sklearn.neural_network 
import sklearn.linear_model
import time

xs = np.array([
    0, 0,
    0, 1,
    1, 0,
    1, 1
]).reshape(4, 2)  # pour créer un tableau de 4 entrées (chacune contenant 2 valeurs)

print("inputs:\n", xs)

ys = np.array([0, 1, 1, 0]).reshape(4,)  # 4 outputs (chacune contenant 1 valeur)
print("wanted outputs:\n", ys)

model = sklearn.neural_network.MLPClassifier(
    activation='logistic', learning_rate_init=0.01, 
    max_iter=1000, hidden_layer_sizes=(3,), verbose=True) # logistic sigmoid
print(model.get_params())
x = time.time()
model.fit(xs, ys) 
y = time.time()
print('score:', model.score(xs, ys))
print('predictions:', model.predict(xs))
print('expected:', np.array([0, 1, 1, 0]))

print("Execution time: ", y-x)