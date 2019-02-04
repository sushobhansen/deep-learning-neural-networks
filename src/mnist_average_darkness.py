from collections import defaultdict
import mnist_loader

'''
A naive classifier based on the average darkness of the image
Essentially a nearest-neighbor classifier based on which training data had the closest
average darkness to the test data
'''

def main():
	training_data, validation_data, test_data = mnist_loader.load_data()

	#Evaluate average darkness of training data
	avgs = average_darkness(training_data)

	#Test how many test data are classified correctly
	num_correct = sum(int(guess_digit(image,avgs) == digit) for image,digit in zip(test_data[0],test_data[1]))
	print('Naive classifier: ',num_correct,' of ',len(test_data[1]),' correct')

def average_darkness(training_data):
	'''
	Returns a dict whose key is a digit (0-9) and value is the average darkness of
	training images containing the digit
	'''
	digit_counts = defaultdictI(int)
	darkness = defaultdict(float)

	for image, digit in zip(training_data[0], training_data[1]):
		digit_counts[digit] += 1
		darkness[digit] += sum(image)

	avgs = defaultdict(float)

	for digit,n in digit_counts.items():
		avgs[digit] = darkness[digit]/n

	return avgs

def guess_digit(image, avgs):
	'''
	Nearest neighbor estimator
	'''
	darkness = sum(image)
	distances = {k: abs(v-darkness) for k,v in avgs.items()}
	return min(distances, key=distances.get)

if __name__== '__main__':
	main()