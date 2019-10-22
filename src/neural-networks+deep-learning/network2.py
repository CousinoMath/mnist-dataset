"""
network2.py
~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.
"""

### Libraries
# Standard libraries
import json
import random
import sys

# Third-party libraries
import numpy as np

def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """The derivative of the sigmoid function."""
    sz = sigmoid(z)
    return sz * (1 - sz)

def coordinate_vector(dim, k):
    """Returns the ``k``th ``dim``-dimensional coordinate vector."""
    e = np.zeros((dim, 1))
    e[k] = 1.0
    return e

def load(filename):
    """Loads a neural network from the file ``filename``. Returns an instance
    of Network."""
    fp = open(filename, 'r')
    data = json.load(fp)
    fp.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.biases = [np.array(b) for b in data["biases"]]
    net.weights = [np.array(w) for w in data["weights"]]
    return net

### Define the quadratic and cross-entropy cost functions
class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        """:a: predicted output
        :y: observed output
        :returns: the cost associated betweeen a predicted output  ``a`` and
        observed output ``y``"""
        return 0.5 * np.linalg.norm(a - y) ** 2
    
    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y) * sigmoid_prime(z)
    
class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        """:a: predicted output
        :y: observed output
        :returns: the cost associated betweeen a predicted output  ``a`` and
        observed output ``y``"""
        # np.nan_to_num ensures that 0.0 * log 0.0 is converted (correctly) to
        # 0.0.
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))
    
    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y)

class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        """:sizes: a list of number of neurons in the respective layers of the
        network
        :returns: a Network object with biases initialized as standard normal
        deviates, and weights initialized as standardized normal deviates with
        standard deviation = 1/\\sqrt{\\text{number of neurons connecting to the
        same neuron}}"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost
    
    def default_weight_initializer(self):
        """Initialize each weight via a Gaussian with mean 0 and standard
        deviation equal to the inverse square root of the number of weights
        connecting to the same neuron. Initialize the biases using a standard
        normal distribution."""
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    
    def large_weight_initializer(self):
        """Initialized each bias and weight with a standard normal deviate."""
        self.biases = [np.random.rand(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.rand(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    

    def feed_forward(self, a):
        """Returns the output of the network on input ``a``."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def stochastic_gradient_descent(self, training_data, epochs,
                                    mini_batch_size, eta, lambda_ = 0.0,
                                    evaluation_data = None,
                                    monitor_evaluation_cost = False,
                                    monitor_evaluation_accuracy = False,
                                    monitor_training_cost = False,
                                    monitor_training_accuracy = False):
        """Use stochastic gradient descent to train the neural network. We
        can monitor the cost and accuracy on either the evaluation data or the
        training data, by setting the appropriate flags.
        :training_data: list of tuples ``(x, y)`` representing inputs and
        outputs
        :epochs: number of epochs to run through
        :mini_batch_size: size of each mini-batch
        :eta: learning rate
        :lambda_: regularization parameter
        :evaluation_data: usually either validation or testing data
        :returns: a tuple of evaluation cost & accuracy and training cost & 
        accuracy"""
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lambda_, n)
            print("Epoch {} training complete".format(j))
            if monitor_training_cost:
                cost = self.total_cost(training_data, lambda_)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lambda_)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data, convert=False)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy
    
    def update_mini_batch(self, mini_batch, eta, lambda_, n):
        """Update network's weights and biases by gradient descent via
        backpropagation to a single mini-batch.
        :mini_batch: list of tuples ``(x, y)`` representing inputs and outputs
        :eta: learning rate
        :lambda_: regularization parameter
        :n: total size of the training data"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        w_coeff = 1 - eta * (lambda_ / n)
        nabla_coeff = eta / len(mini_batch)
        self.biases = [b - nabla_coeff * nb
                        for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w_coeff * w - nabla_coeff * nw
                        for w, nw in zip(self.weights, nabla_w)]
    
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the gradient of
        the cost function C_x. ``nabla_b`` and ``nabla_w`` are layer-by-layer
        lists of numpy arrays similar to ``self.biases`` and ``self.weights``,
        respectively"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feed-forward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            dleta = np.dot(self.weights[-l + 1].tranpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)
    
    def accuracy(self, data, convert=False, output_fn=np.argmax):
        """Return the number of inputs in ``data`` for which the neural network
        outputs the correct result.
        :convert: should be set to ``False`` if data set is validation or test
        data (the usual case), and to ``True`` if the data set is the training
        data.
        :output_fn: is the output function to be applied to the network's
        output."""
        if convert:
            results = [(output_fn(self.feed_forward(x)), output_fn(y))
                        for (x, y) in data]
        else:
            results = [(output_fn(self.feed_forward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
    
    def total_cost(self, data, lambda_, convert=False):
        """Return the total cost for the data set ``data``.
        :convert: should be set to ``False`` if the data set is validation or
        testing data (the usual case), and to ``True`` if the data set is the
        training data."""
        cost = 0.0
        n = len(data)
        for x, y in data:
            a = self.feed_forward(x)
            if convert: y = coordinate_vector(10, y)
            cost += (self.cost).fn(a, y) / n
        cost += 0.5 * (lambda_/n)*sum(np.linalg.norm(w) ** 2
                                        for w in self.weights)
        return cost
    
    def save(self, filename):
        """Save the neural network to a JSON file ``filename``."""
        data = { "sizes": self.sizes,
                "biases": [b.tolist() for b in self.biases],
                "weights": [w.tolist() for w in self.weights],
                "cost": str((self.cost).__name__) }
        fp = open(filename, 'w')
        json.dump(data, fp)
        fp.close()

