import numpy as np

class Network:

	#The entire neural network is contained in this class
	#The network is expressed in the form of a number of neurons and their weights and biases in each layer
	#The size (no of neurons) in each layer is required for initialization as a list sizes
	#Biases and weights and initialized at random
	
	def __init__(self, sizes):
		self.num_layers = len(sizes) #number of layers (including input and output layers)
		self.sizes = sizes #number of neurons per layer (including input and output layers)
		self.biases = [np.random.randn(y,1) for y in sizes[1:]] #biases of each neuron, except in the first layer
		self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])] #weights between previous x and current y layer of neurons (note it's indexed as (y,x))

