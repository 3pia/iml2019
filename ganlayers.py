import tensorflow as tf
from tensorflow.python.layers.utils import deconv_output_length
import numpy as np


def spectral_norm(W, use_gamma=False, factor=None, name='sn'):
    shape = W.get_shape().as_list()
    if len(shape) == 1:
        sigma = tf.reduce_max(tf.abs(W))
    else:
        if len(shape) == 4:
            _W = tf.reshape(W, (-1, shape[3]))
            shape = (shape[0] * shape[1] * shape[2], shape[3])
        elif len(shape) == 5:
            _W = tf.reshape(W, (-1, shape[4]))
            shape = (shape[0] * shape[1] * shape[2] * shape[3], shape[4])
        else:
            _W = W
        u = tf.get_variable(
            name=name + "_u",
            shape=(1, shape[0]),
            initializer=tf.random_normal_initializer,
            trainable=False
        )

        _u = u
        for _ in range(1):
            _v = tf.nn.l2_normalize(tf.matmul(_u, _W), 1)
            _u = tf.nn.l2_normalize(tf.matmul(_v, tf.transpose(_W)), 1)
        _u = tf.stop_gradient(_u)
        _v = tf.stop_gradient(_v)
        sigma = tf.reduce_mean(tf.reduce_sum(_u * tf.transpose(tf.matmul(_W, tf.transpose(_v))), 1))
        update_u_op = tf.assign(u, _u)
        with tf.control_dependencies([update_u_op]):
            sigma = tf.identity(sigma)

    if factor:
        sigma = sigma / factor

    if use_gamma:
        s = tf.svd(tf.transpose(_W), compute_uv=False)[0]
        gamma = tf.get_variable(name=name + "_gamma", initializer=s)
        return gamma * W / sigma
    else:
        return W / sigma


def _conv_sn(conv, inputs, filters, kernel_size, name,
             strides=1,
             padding='valid',
             activation=None,
             use_bias=True,
             kernel_initializer=tf.glorot_uniform_initializer(),
             bias_initializer=tf.zeros_initializer(),
             use_gamma=False,
             factor=None, transposed=False):
    input_shape = inputs.get_shape().as_list()
    c_axis, h_axis, w_axis = 3, 1, 2  # channels last
    input_dim = input_shape[c_axis]
    with tf.variable_scope(name):
        if transposed is True:
            kernel_shape = kernel_size + (filters, input_dim)
            kernel = tf.get_variable('kernel', shape=kernel_shape, initializer=kernel_initializer)
            height, width = input_shape[h_axis], input_shape[w_axis]
            kernel_h, kernel_w = kernel_size
            stride_h, stride_w = strides
            out_height = deconv_output_length(height, kernel_h, padding, stride_h)
            out_width = deconv_output_length(width, kernel_w, padding, stride_w)
            output_shape = (input_shape[0], out_height, out_width, filters)
            outputs = conv(inputs, spectral_norm(kernel, use_gamma=use_gamma, factor=factor), tf.stack(output_shape), strides=(1, *strides, 1), padding=padding.upper())
        else:
            kernel_shape = kernel_size + (input_dim, filters)
            kernel = tf.get_variable('kernel', shape=kernel_shape, initializer=kernel_initializer)
            outputs = conv(inputs, spectral_norm(kernel, use_gamma=use_gamma, factor=factor), strides=(1, *strides, 1), padding=padding.upper())
        if use_bias is True:
            bias = tf.get_variable('bias', shape=(filters,), initializer=bias_initializer)
            outputs = tf.nn.bias_add(outputs, bias)
        if activation is not None:
            outputs = activation(outputs)

    return outputs


def dense_sn(inputs, units, name,
             activation=None,
             use_bias=True,
             kernel_initializer=tf.glorot_uniform_initializer(),
             bias_initializer=tf.zeros_initializer(),
             use_gamma=False,
             factor=None):

    input_shape = inputs.get_shape().as_list()

    with tf.variable_scope(name):
        kernel = tf.get_variable('kernel', shape=(input_shape[-1], units), initializer=kernel_initializer)
        outputs = tf.matmul(inputs, spectral_norm(kernel, use_gamma=use_gamma, factor=factor))
        if use_bias is True:
            bias = tf.get_variable('bias', shape=(units,), initializer=bias_initializer)
            outputs = tf.nn.bias_add(outputs, bias)
        if activation is not None:
            outputs = activation(outputs)

    return outputs


def conv2d_sn(inputs, filters, kernel_size, name,
              strides=(1, 1),
              padding='valid',
              activation=None,
              use_bias=True,
              kernel_initializer=tf.glorot_uniform_initializer(),
              bias_initializer=tf.zeros_initializer(),
              use_gamma=False,
              factor=None):
    return _conv_sn(tf.nn.conv2d, inputs, filters, kernel_size, name,
                    strides=strides,
                    padding=padding,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    use_gamma=use_gamma,
                    factor=factor)


def conv2d_transpose_sn(inputs, filters, kernel_size, name,
                        strides=(1, 1),
                        padding='valid',
                        activation=None,
                        use_bias=True,
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        use_gamma=False,
                        factor=None):

    return _conv_sn(tf.nn.conv2d_transpose, inputs, filters, kernel_size, name,
                    strides=strides,
                    padding=padding,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    use_gamma=use_gamma,
                    factor=factor, transposed=True)