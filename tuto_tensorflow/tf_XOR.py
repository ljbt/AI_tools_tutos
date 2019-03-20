import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))

x = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
y = np.array([[1], [1], [0], [0]])

# define placeholders for input x and output y
tf_features = tf.placeholder(tf.float32, [None, 2])    # x
tf_targets = tf.placeholder(tf.float32, [None, 1])     # y

# init weights
# 5 hidden nodes
w1 = init_weights([2, 3])
b1 = tf.Variable(1.0, [3])
w2 = init_weights([3, 1])
b2 = tf.Variable(1.0, [1])

# first layer
z1 = tf.matmul(tf_features, w1) + b1
a1 = tf.nn.sigmoid(z1)

# output layer
z2 = tf.matmul(a1, w2) + b2
py = tf.nn.sigmoid(z2)

# init learning rate
lr = 0.01
# init epochs
epochs = 50000

# init cost function
cost = tf.reduce_mean(tf.square(py-tf_targets))

# train function and optimizer
optimizer = tf.train.GradientDescentOptimizer(lr)
train = optimizer.minimize(cost)  #opération d'entrainement qui minimise le cout (avec la méthode de descente de gradient)
#train = tf.train.AdamOptimizer(lr).minimize(cost)

# save costs for plotting
costs = []

# create session and init variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

start=time.time()
# start training
for i in range(epochs):
    sess.run(train, feed_dict={tf_features: x, tf_targets: y})

    c = sess.run(cost, feed_dict={tf_features: x, tf_targets: y})
    costs.append(c)

    if i % 5000 == 0:
        print("Epoch: ", i, " Cost: ", c)

print("Training complete.")
end=time.time()
print("Time :", end-start)

# make prediction
correct_prediction = tf.equal(tf.round(py), tf_targets) #définition d'une bonne prédiction
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("Theo :\n", sess.run(tf_targets, feed_dict={tf_targets: y}))
print("Reel :\n", sess.run(py, feed_dict={tf_features: x, tf_targets: y}))
print("Accuracy :", sess.run(accuracy, feed_dict={tf_features: x, tf_targets: y}))

# plot cost
plt.plot(costs)
plt.show()



