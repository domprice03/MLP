import numpy as np
import random


class MLP(object):
    def __init__(self, layers, x_train, y_train, x_test, y_test):
        self.layers = layers
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.activations = []
        self.z_values = []
        for i in range(len(layers)):
            self.activations.append(np.zeros(layers[i]))
            self.z_values.append(np.zeros(layers[i]))

        self.biases = []
        for i in range(len(layers)-1):
            self.biases.append(np.random.randn(layers[i+1], 1))
        self.weights = []
        for i in range(len(layers)-1):
            self.weights.append(np.random.randn(layers[i+1], layers[i]))

        self.bias_derivatives = []
        self.weight_derivatives = []

        self.sum_errors = 0

    def initialise_derivatives(self):
        bias_derivatives = []
        for b in self.biases:
            bias_derivatives.append(np.zeros(b.shape))
        weight_derivatives = []
        for w in self.weights:
            weight_derivatives.append(np.zeros(w.shape))
        return bias_derivatives, weight_derivatives

    def forward_propagate(self, a):
        self.activations = [a, ]
        self.z_values = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a)+b
            a = self.sigmoid(z)
            self.z_values.append(z)
            self.activations.append(a)
        return a

    def backpropagation(self, x, y):
        bias_derivatives, weight_derivatives = self.initialise_derivatives()
        self.forward_propagate(x)
        delta = self.cost_derivative(self.activations[-1], y) * self.sigmoid_derivative(self.z_values[-1])
        bias_derivatives[-1] = delta
        weight_derivatives[-1] = np.dot(delta, self.activations[-2].T)
        for i in range(len(self.layers)-3, -1, -1):
            z = self.z_values[i]
            delta = np.dot(self.weights[i + 1].T, delta) * self.sigmoid_derivative(z)
            bias_derivatives[i] = delta
            weight_derivatives[i] = np.dot(delta, self.activations[i].T)
        # Stores the sum of all derivatives so they can be averaged later when updating weights and biases
        for i, dnb in enumerate(bias_derivatives):
            self.bias_derivatives[i] += dnb
        for i, dnw in enumerate(weight_derivatives):
            self.weight_derivatives[i] += dnw
        self.sum_errors += self.mse(y, self.activations[-1])

    def stochastic_gradient_descent(self, epochs, mini_batch_size, learning_rate):
        for i in range(epochs):
            self.sum_errors = 0
            # Create random mini batches from x and y training data.
            training_data = list(zip(self.x_train, self.y_train))
            random.shuffle(training_data)
            mini_batches = []
            for j in range(0, len(training_data), mini_batch_size):
                mini_batches.append(training_data[j:j+mini_batch_size])
            # Apply a gradient descent step for each mini batch
            for mini_batch in mini_batches:
                self.bias_derivatives, self.weight_derivatives = self.initialise_derivatives()
                for x, y in mini_batch:
                    self.backpropagation(x, y)
                for k, (w, wd) in enumerate(zip(self.weights, self.weight_derivatives)):
                    self.weights[k] = w - (learning_rate / len(mini_batch)) * wd
                for k, (b, bd) in enumerate(zip(self.biases, self.bias_derivatives)):
                    self.biases[k] = b - (learning_rate / len(mini_batch)) * bd
            print(f'Error: {self.sum_errors / len(self.x_train)} at epoch {i+1}')
        self.evaluate()

    def evaluate(self):
        test_results = []
        correct = 0
        for x, y in list(zip(self.x_test, self.y_test)):
            output = np.argmax(self.forward_propagate(x))
            target = y
            test_results.append((output, target))
        for (x, y) in test_results:
            if int(x) == int(y):
                correct += 1
        print(f'Evaluating test data: {correct} / {len(self.x_test)}')

    def mse(self, target, output):
        # Averages all elements of an array of values (between 0 and 1) squared
        return np.average((target - output) ** 2)

    def cost_derivative(self, output_activations, y):
        # With respect to activation
        return 2 * (output_activations - y)

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
