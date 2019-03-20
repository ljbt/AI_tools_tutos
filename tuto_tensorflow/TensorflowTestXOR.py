import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def get_dataset():
	"""
		Method used to generate the data
	"""
	#Numbers of row per class
	row_per_class = 100
	#Generate rows
	sick = np.random.randn(row_per_class, 2) + np.array([-2, -2])
	sick_2 = np.random.randn(row_per_class, 2) + np.array([2, 2])
	
	healthy = np.random.randn(row_per_class, 2) + np.array([-2, 2])
	healthy_2 = np.random.randn(row_per_class, 2) + np.array([2, -2])
	
	#features : personnes
	features = np.vstack([sick, sick_2, healthy, healthy_2])
	#targets : malade ou non
	targets = np.concatenate((np.zeros(row_per_class * 2), np.zeros(row_per_class * 2) + 1))

	targets = targets.reshape(-1, 1)

	return features, targets
 
if __name__ == '__main__':
	features, targets = get_dataset()
	
    #déclaration
	tf_features = tf.placeholder(tf.float32, shape=[None, 2])	#chaque personne a deux caractéristiques
	tf_targets = tf.placeholder(tf.float32, shape=[None,1])		#1 seule valeur : malade ou non
	
	#First
	w1 = tf.Variable(tf.random.normal([2, 3]))      #poids  #pour chacune des caractéristiques il va y avoir 3 poids associés
	b1 = tf.Variable(tf.zeros([3]))                 #biais  #trois neurones en couche cachée
	#Operations
	z1 = tf.matmul(tf_features, w1) + b1            #pré-activation f*w+b
	a1 = tf.nn.sigmoid(z1)
	
	#Output neuron
	w2 = tf.Variable(tf.random.normal([3, 1]))
	b2 = tf.Variable(tf.zeros([1]))	                #un neurone en sortie
	#Operations
	z2 = tf.matmul(a1, w2) + b2
	py = tf.nn.sigmoid(z2)                          #activation du neurone (résultat)
	
	cost = tf.reduce_mean(tf.square(py-tf_targets)) #erreur globale : moyenne (py-tf_targets)²
	
	correct_prediction = tf.equal(tf.round(py), tf_targets) #définition d'une bonne prédiction
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
	train = optimizer.minimize(cost)                #opération d'entrainement qui minimise le cout (avec la méthode de descente de gradient)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())     #initialise toutes les variables
	
	for e in range(10000):      #plus on fait d'epoch plus le réseau va apprendre et mieux classifier
		
		sess.run(train, feed_dict={
			tf_features: features,      #définition des entrées
			tf_targets: targets
		})

		print("accuracy =", sess.run(accuracy, feed_dict={
			tf_features: features,
			tf_targets: targets
		}))

