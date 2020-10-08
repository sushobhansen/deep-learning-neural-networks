import mnist_loader
import network
import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#net = network.Network([784, 50, 10])
#net.SGD(list(training_data), 30, 10, 3.0, test_data=list(test_data))
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(list(training_data), 30, 10, 0.5, evaluation_data=list(test_data),monitor_evaluation_accuracy=True)
