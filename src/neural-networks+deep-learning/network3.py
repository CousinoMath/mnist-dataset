"""
network3.py
~~~~~~~~~~~

A Theano-based program for training and running simple neural
networks.
Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).
When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.
Because the code is based on Theano, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.
This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably,
`http://deeplearning.net/tutorial/lenet.html`_ ), from Misha Denil's
implementation of dropout (`https://github.com/mdenil/dropout`_ ), and
from Chris Olah (`http://colah.github.io`_ ).
Written for Theano 0.6 and 0.7, needs some changes for more recent
versions of Theano.
"""

### Libraries
# Standard libraries
import _pickle
import gzip

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet import conv
import theano.tensor.nnet import softmax
import theano.tensor import shared_randomstreams
import theano.tensor.signal import downsample

def linear(z): return z
def reLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

### Constants
GPU = False
if GPU:
    print("""Trying to run under a GPU. If this is not desired, then modify
        If this is not desired, then modify ``network3.py`` to set the GPU flag.
        to False.""")
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print("""Running with a CPU. If this is not desired, then modify
        ``network3.py`` to set the GPU flag to True""")

### Load the MNIST data
def load_data_shared(filename='../../data/mnist.pkl.gz'):
    fp = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = _pickle.load(fp)
    fp.close()
    def shared(data):
        """Place the data into shared variables. This allows Theano to copy the
        data to the GPU, if one is available."""
        shared_x = theano.shared(
                    np.asarray(data[0], dtype=theano.config.floatX),
                    borrow=True)
        shared_y = theano.shared(
                    np.asarray(data[1], dtype=theano.config.floatX),
                    borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]

### Main class used to construct and train networks
class Network(object):
    def __init__(self, layers, mini_batch_size):
        """Takes in a list of ``layers``, describing the network architecture,
        and a value for the ``mini_batch_size`` to be used during training by
        stochastic gradient descent."""
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_input(self.x self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j - 1], self.layers[j]
            layer.set_input(prev_layer.output, prev_layer.output_dropout,
                            self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
    
    def stochastic_gradient_descent(self, training_data, epochs,
                                    eta, validation_data, test_data,
                                    lambda_ = 0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data
        batch_size = self.mini_batch_size

        # compute umber of mini-batches for training, validation, and testing
        num_training_batches = size(training_data) / batch_size
        num_validation_batches = size(validation_data) / batch_size
        num_test_batches = size(test_data) / batch_size
        l2_norm_squared = sum([(layer.w ** 2).sum() for layer in self.layer])
        cost = self.layers[-1].cost(self) + \
            0.5 * lambda_ * l2_norm_squared / num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param - eta * grad)
                    for param, grad in zip(self.params, grads)]
        
        # define functions to train a mini-batch, and to compute the accuracy
        # in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i * batch_size:(i + 1) * batch_size],
                self.y:
                training_y[i * batch_size:(i + 1) * batch_size],
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i * batch_size:(i + 1) * batch_size],
                self.y:
                validation_y[i * batch_size:(i + 1) * batch_size],
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i * batch_size:(i + 1) * batch_size],
                self.y:
                test_y[i * batch_size:(i + 1) * batch_size],
            })
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i * batch_size:(i + 1) * batch_size]
            })
        # Do the actual training
        best_validation_accuracy = 0.0
        for epoch in range(epochs):
            for mini_batch_index in range(num_training_batches):
                iteration = num_training_batches * epoch + mini_batch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(mini_batch_index)
                if (iteration + 1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j)
                            for j in range(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy so far.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j)
                                    for j in range(num_test_batches)])
                            print("The corresponding test accuracy is {0:.2%}".format(test_accuracy))
        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

### Define layer types
