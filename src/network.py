import numpy as np
import random

class Network:

	'''
	The entire neural network is contained in this class
	The network is expressed in the form of a number of neurons and their weights and biases in each layer
	The size (no of neurons) in each layer is required for initialization as a list sizes
	'''
	
	def __init__(self, sizes):

		'''
		Initializes the neural network by taking in a list of the number of neurons per layer (including input and output layers)
		Biases and weights are assigned at random
		'''

		self.num_layers = len(sizes) #number of layers (including input and output layers)
		self.sizes = sizes #number of neurons per layer (including input and output layers)
		self.biases = [np.random.randn(y,1) for y in sizes[1:]] #biases of each neuron, except in the first layer
		self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])] #weights between previous x and current y layer of neurons (note it's indexed as (y,x))
		
	def feedforward(self, a):
		'''
		Takes an input a and returns an output from the sigmoid function after running through the entire network
		Note that this function rewrites a copy of the input a internally
		'''
		for b, w in zip(self.biases,self.weights):
			a = sigmoid(np.dot(w,a)+b)
		
		return a
		
	def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
		'''
		Performs a stochastic gradient descent on the training data to tune the weights and biases
		A mini-batch SGD is used to sample the data to tune it, and this is done for several epochs
		training_data is a tuple (x,y) of inputs and desired outputs
		eta is the gradient descent step size
		If test_data is provided, then the function analyzed the test data and prints output
		'''
		
		if test_data: n_test = len(test_data)
		
		n = len(training_data)
		
		for j in range(epochs):
			random.shuffle(training_data) #shuffle the training data 
			
			#Create mini-batches by sampling the shuffled training data
			mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
			
			for mini_batch in mini-batches:
				self.update_mini_batch(mini_batch, eta) #Gradient descent on the mini-batch
			
			if test_data:
				print('Epoch ',j,': ',self.evaluate(test_data),' / ',n_test)
			else:
				print('Epoch ',j,' complete')

#Define sigmoid function
def sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))