import tensorflow as tf
import utils

def fc_layer(x, shape, name):
    num_inputs, num_outputs = shape
    W = utils.weight_variable(shape, 1.0, name + "/W")
    b = utils.bias_variable([num_outputs], 0.0, name + "/b")
    return tf.nn.sigmoid(tf.matmul(x, W) + b)

def weight_variable(shape, std, name):
    initial = tf.truncated_normal(shape, stddev = std)   #生成正太分布数据，通常作为权重的初始值
    W = tf.Variable(initial, name = name)
    return W

def bias_variable(shape, value, name):
	initial = tf.constant(value, shape = shape)
	b = tf.Variable(initial, name = name)
	return b

def conv_layer(x, filter_shape, stride, sigmoid, name):
    filter_width, num_inputs, num_outputs = filter_shape
    W = weight_variable(filter_shape, 0.1, name + "/W")
    b = bias_variable([num_outputs], 0.0, name + "/b")
    z = tf.nn.conv1d(x, W, stride = stride, padding = 'SAME') + b
    a = tf.nn.sigmoid(z) if sigmoid else tf.nn.tanh(z)
    return a
