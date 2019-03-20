import numpy as np
import theano
import theano.tensor as T
import time

# Set inputs and correct output values
inputs = [[0,0], [1,1], [0,1], [1,0]]
outputs = [0, 0, 1, 1]
 
# Set training parameters
alpha = 0.01 # Learning rate
training_iterations = 50000
hidden_layer_nodes = 3
 
# Define tensors, poids et biais
x = T.matrix("x")
y = T.vector("y")
b1 = theano.shared(value=1.0, name='b1')
b2 = theano.shared(value=1.0, name='b2')
 
# Set random seed
rng = np.random.RandomState(2345)
 
# Initialize weights
w1_array = np.asarray(rng.uniform(low=-1, high=1, size=(2, hidden_layer_nodes)),
                      dtype=theano.config.floatX) # Force type to 32bit float for GPU
w1 = theano.shared(value=w1_array, name='w1')
 
w2_array = np.asarray(rng.uniform(low=-1, high=1, size=(hidden_layer_nodes, 1)),
                      dtype=theano.config.floatX) # Force type to 32bit float for GPU
w2 = theano.shared(value=w2_array, name='w2')
 
a1 = T.nnet.sigmoid(T.dot(x, w1) + b1)  # Input -> Hidden #produit scalaire
a2 = T.nnet.sigmoid(T.dot(a1, w2) + b2) # Hidden -> Output
hypothesis = T.flatten(a2) 
# Il faut laplatir pour que les hypotheses (matrice) et y (vecteur) aient la meme forme.
 
cost = T.sum((y - hypothesis) ** 2) # calcul de lerreur quadratic 
 
updates_rules = [ #mise a jour des poids/biais du reseau
    (w1, w1 - alpha * T.grad(cost, wrt=w1)),
    (w2, w2 - alpha * T.grad(cost, wrt=w2)),
    (b1, b1 - alpha * T.grad(cost, wrt=b1)),
    (b2, b2 - alpha * T.grad(cost, wrt=b2))
]
 
# Theano compiled functions, execution 
train = theano.function(inputs=[x, y], outputs=[hypothesis, cost], updates=updates_rules)
predict = theano.function(inputs=[x], outputs=[hypothesis])
 
# Training
cost_history = []

start = time.time()
for i in range(training_iterations):
    if (i+1) % 5000 == 0:
        print "Iteration #%s: " % str(i+1)
        print "Cost: %s" % str(cost)
    h, cost = train(inputs, outputs)
    cost_history.append(cost)
end = time.time()

test_data = [[0,0], [1,1], [0,1], [1,0]]
predictions = predict(test_data)
print np.round(predictions)

print('Time (s):', end - start)
