import tensorflow as tf
from net.utils import add_variable_summary


def conv(input_layer, filters, kernel_size=[3, 3], activation=tf.nn.relu):
    """Convolutional layer with included tensorboard summary.

    Arguments:
        input_layer {tensor} -- Input layer or image
        filters {integer} -- the dimensionality of the output space

    Keyword Arguments:
        kernel_size {list} -- specify the height and width of the 2D convolution window (default: {[3,3]})
        activation {function} -- Activation function to use (default: {tf.nn.relu})
    """
    layer = tf.layers.conv2d(
        inputs=input_layer,
        filters=filters,
        kernel_size=kernel_size,
        activation=activation
    )
    add_variable_summary(layer, 'convolution')

    return layer


def max_pool(input_layer, pool_size=[2, 2], strides=2):
    """Max pooling with included tensorboard summary.

    Arguments:
        input_layer {tensor} -- Must have rank 4 (e.g. convolutional layer)

    Keyword Arguments:
        pool_size {list} -- size of the window (default: {[2,2]})
        strides {int} -- amount of strides the kernel should slide (default: {2})
    """
    layer = tf.layers.max_pooling2d(
        inputs=input_layer,
        pool_size=pool_size,
        strides=strides
    )
    add_variable_summary(layer, 'pooling')

    return layer


def fc(input_layer, units, activation=tf.nn.relu):
    """Functional interface for the densely-connected layer.

    - take any vector of single dimension and map to any number of hidden units.
    - outputs = activation(inputs * kernel + bias)

    Arguments:
        input_layer {tensor} --
        units {integer} -- dimensionality of the output space

    Keyword Arguments:
        activation {function} -- Activation function to use (default: {tf.nn.relu})

    Returns:
        tensor -- Output tensor the same shape as inputs except the last dimension is of size units.
    """
    layer = tf.layers.dense(
        inputs=input_layer,
        units=units,
        activation=activation
    )
    add_variable_summary(layer, 'dense')

    return layer
