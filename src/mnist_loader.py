import cPickle
import gzip
import numpy as np 

def load_data():
	'''
	Loads MNIST data from zip file and returns training data, validation data, and test data
	'''

	f = gzip.open('../data/mnist.pkl.gz','rb')
	training_data, validation_data, test_data = cPickle.load(f)
	f.close()

	return (training_data, validation_data, test_data)

def load_data_wrapper():
	'''
	Similar to load_data, except that the second element of training output tuple is a unit vector of size 10
	corresponding to the appropriate label (0-9). The first element is also reshaped into (n,1)
	'''

	tr_d, va_d, te_d = load_data()

	training_inputs = [np.reshape(x,(784,1)) for x in tr_d[0]]
	training_results = [vectorized_result(y) for y in tr_d[1]]
	training_data = zip(training_inputs,training_results)

	validation_inputs = [np.reshape(x,(784,1)) for x in va_d[0]]
	validation_data = zip(validation_inputs,va_d[1])

	test_inputs = [np.reshape(x,(784,1)) for x in te_d[0]]
	test_data = zip(test_inputs,te_d[1])

	return (training_data, validation_data, test_data)

def vectorized_result(j):
	'''
	Returns the label in the form of a unit vector
	'''
	e = np.zeros((10,1))
	e[j] = 1.0
	return e