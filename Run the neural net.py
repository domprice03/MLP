import tensorflow as tf
import numpy as np
import NeuralNet

digits = tf.keras.datasets.mnist.load_data(path="mnist.npz")
training_data = digits[0]
test_data = digits[1]

training_data_x = training_data[0]
x_train = []
for i, x in enumerate(training_data_x):
    x_train.append(np.array([x.flatten()/255]).T)
training_data_y = training_data[1]

y_train = []
for i, y in enumerate(training_data_y):
    array = np.zeros((10, 1))
    array[y] = 1
    y_train.append(array)

test_data_x = test_data[0]
x_test = []
for i, x in enumerate(test_data_x):
    x_test.append(np.array([x.flatten()/255]).T)

y_test = test_data[1]


def main():
    neural_net = NeuralNet.MLP([784, 100, 10], x_train, y_train, x_test, y_test)
    neural_net.stochastic_gradient_descent(30, 10, 2.8)


if __name__ == '__main__':
    main()
