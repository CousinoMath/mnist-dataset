"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    sz = sigmoid(z)
    return sz * (1.0 - sz)

class Network(object):
    def __init__(self, sizes):
        """Initialize neural network, biases and weights for the
        network are initialized randomly, using standard normal
        deviates.

        :param sizes: contains the number of neurons in the
        respective layers of the network."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feed_forward(self, a):
        """Returns the output of the network if ``a`` is the input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def evaluate(self, test_data, output_fn=np.argmax):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(output_fn(self.feed_forward(x)), output_fn(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \\partial C_x /
        \\partial a for the output activations."""
        return (output_activations - y)

    def stochastic_gradient_descent(self, training_data, epochs,
            mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.
        
        :param training_data: list of tuples ``(x, y)`` representing the
        training inputs and the desired outputs.
        :param epochs: number of epochs
        :param mini_batch_size: size of mini-batches
        :param eta: learning rate
        :param test_data: test data against which the network will be
        evaluated."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j,
                    self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
    
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.

        :param mini_batch: list of tuples ``(x, y)``
        :param eta: the learning rate"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb
                for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw
                for nw, dnw in zip(nabla_w, delta_nabla_w)]
        nabla_coeff = eta / len(mini_batch)
        self.biases = [b - nabla_coeff * nb
                        for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - nabla_coeff * nw
                        for w, nw in zip(self.weights, nabla_w)]
    
    def backprop(self, x, y):
        """Returns a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.
        
        :param x: input activations
        :param y: output activations
        :returns: a pair of layer-by-layer lists of numpy arrays,
        similar to ``self.biases`` and ``self.weights``"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feed-forward
        activation = x
        # list to store all activations, layer-by-layer
        activations = [x]
        # list to store all the ``z`` vectors, layer-by-layer
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backwards pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)