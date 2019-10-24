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
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal.pool import pool_2d

def linear(z): return z
def reLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

### Constants
GPU = False
if GPU:
    print("""Trying to run under a GPU. If this is not desired, then 
        modify ``network3.py`` to set the GPU flag to False.""")
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print("""Running with a CPU. If this is not desired, then modify
        ``network3.py`` to set the GPU flag to True.""")

### Load the MNIST data
def load_data_shared(filename="../../data/mnist.pkl.gz"):
    with gzip.open(filename, 'rb') as fp:
        training_data, validation_data, test_data = _pickle.load(fp)
    def shared(data):
        """Place the data into shared variables. This allows Theano to
        copy the data to the GPU, if one is available."""
        shared_x = theano.shared(
                    np.asarray(data[0], dtype=theano.config.floatX),
                    borrow=True)
        shared_y = theano.shared(
                    np.asarray(data[1], dtype=theano.config.floatX),
                    borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data),
        shared(test_data)]

### Miscellanea
def size(data):
    "Return the size of the dataset ``data``."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n = 1, p = 1 - p_dropout, size=layer.shape)
    return layer * T.cast(mask, theano.config.floatX)

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
        init_layer.set_input(self.x, self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j - 1], self.layers[j]
            layer.set_input(prev_layer.output, prev_layer.output_dropout,
                            self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
    
    def stochastic_gradient_descent(self, training_data, epochs,
                                    mini_batch_size, eta,
                                    test_data, lambda_ = 0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        test_x, test_y = test_data
        batch_size = self.mini_batch_size

        # compute umber of mini-batches for training, validation, and testing
        num_training_batches = size(training_data) / mini_batch_size
        num_test_batches = size(test_data) / mini_batch_size
        l2_norm_squared = sum([(layer.w ** 2).sum()
                                for layer in self.layers])
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
                training_x[i * self.mini_batch_size:(i + 1) * self.mini_batch_size],
                self.y:
                training_y[i * self.mini_batch_size:(i + 1) * self.mini_batch_size],
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
                iteration = num_training_batches * epoch + \
                    mini_batch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(mini_batch_index)
                if (iteration + 1) % num_training_batches == 0:
                    if test_data:
                        test_accuracy = np.mean(
                            [test_mb_accuracy(j)
                                for j in range(num_test_batches)])
                        print("The test accuracy is {0:.2%}".format(test_accuracy))
        print("Finished training network.")
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

### Define layer types
class ConvolutionalPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them."""

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                    activation_fn=sigmoid):
        """:param filter_shape: is a 4-tuple consisting of the number of
        filters, the number of input feature maps, the filter height,
        and the filter width.
        :param image_shape: is a 4-tuple consisting of the mini-batch size,
        the number of input feature maps, the image height, and the
        image width
        :param poolsize: is a 2-tuple consisting of the ``y`` and ``x``
        pooling sizes"""
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn
        # initialize the weights and biases
        n_out = (filter_shape[0] * \
            np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0,
                    scale=np.sqrt(1/n_out),size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0,
                    size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.b, self.w]
    
    def set_input(self, input, input_dropout, mini_batch_size):
        self.input = input.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=conv_out, filters=self.w,
            filter_shape=self.filter_shape,
            image_shape=self.image_shape)
        pooled_out = pool_2d(
            input=conv_out, ws=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output

class FullyConnectedLayer(object):
    def __init__(self, n_in, n_out, activation_fn=sigmoid,
            p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0.0, scale=np.sqrt(1.0/n_out),
                    size=(n_in,n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.b, self.w]
    
    def set_input(self, input, input_dropout, mini_batch_size):
        self.input = input.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn((1 - self.p_dropout) * \
            T.dot(self.input, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.input_dropout = dropout_layer(
            input_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.input_dropout, self.w) + self.b)
    
    def accuracy(self, y):
        "Return the accuracy of the mini-batch."
        return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer(object):
    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.b, self.w]
    
    def set_input(self, input, input_dropout, mini_batch_size):
        self.input = input.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1 - self.p_dropout) * \
            T.dot(self.input, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.input_dropout = dropout_layer(
            input_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(
            T.dot(self.input_dropout, self.w) + self.b)
    
    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(
            T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])
    
    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))
