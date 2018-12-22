# coding=utf-8
import tensorflow as tf

DEFAULT_WEIGHT_COLLECTION = "weight_set"


def get_full_connect_layer(input, input_num, output_num, namespace, reuse=None, regularization=False, regular_method=tf.nn.l2_loss,
                           collection_name=DEFAULT_WEIGHT_COLLECTION):
    with tf.variable_scope(namespace, reuse=reuse):
        w = tf.get_variable("weights", shape=[input_num, output_num],
                               initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("biases", shape=[output_num], initializer=tf.constant_initializer(0.1))
        z = tf.nn.xw_plus_b(input, w, b)
        if regularization:
            add_to_regulation(w, regular_method, collection_name)
    return z


def add_to_regulation(W, regular_method=tf.nn.l2_loss, collection_name=DEFAULT_WEIGHT_COLLECTION):
    tf.add_to_collection(collection_name, regular_method(W))
    return


def get_regulation(collection_name=DEFAULT_WEIGHT_COLLECTION):
    return tf.add_n(tf.get_collection(collection_name))


def get_conv2d_layer(x, filter_size_length, filter_size_width, in_channels, num_filters, namespace, reuse=None,
                     strides=[1, 1, 1, 1], weight_initialize=tf.truncated_normal_initializer(stddev=0.1),
                     nonlinear=tf.nn.relu, padding="VALID"):
    conv = get_conv2d_layer_without_nonlinear(x, filter_size_length, filter_size_width, in_channels, num_filters,
                                              namespace, reuse, strides, weight_initialize, padding)
    with tf.variable_scope(namespace, reuse=reuse):
        # Apply nonlinearity
        out = nonlinear(conv, name="nonlinear")
    return out


def get_conv2d_layer_without_nonlinear(x, filter_size_length, filter_size_width, in_channels, num_filters, namespace,
                    reuse=None,strides=[1, 1, 1, 1], weight_initialize=tf.truncated_normal_initializer(stddev=0.1), padding="VALID"):
    with tf.variable_scope(namespace, reuse=reuse):
        filter_shape = [filter_size_length, filter_size_width, in_channels, num_filters]
        conv_W = tf.get_variable("weights", shape=filter_shape,
                                 initializer=weight_initialize)
        conv_b = tf.get_variable("biases", shape=[num_filters], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(x, conv_W, strides=strides, padding=padding, name="conv")
        out = tf.nn.bias_add(conv, conv_b, name="linear")
    return out


def get_conv1d_layer(x, filter_size_length, filter_size_width, in_channels, num_filters, namespace, reuse=None, nonlinear=tf.nn.relu, padding="VALID"):
    w_shape = [filter_size_length, filter_size_width, in_channels, num_filters]
    b_shape = [num_filters, filter_size_width]
    with tf.variable_scope(namespace, reuse=reuse):
        input_unstack = tf.unstack(x, axis=2)
        w = tf.get_variable("weights", shape=w_shape,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable("biases", shape=b_shape, initializer=tf.constant_initializer(0.1))
        w_unstack = tf.unstack(w, axis=1)
        b_unstack = tf.unstack(b, axis=1)
        convs = []
        for i in range(len(input_unstack)):
           conv = nonlinear(tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding=padding) + b_unstack[i])
           convs.append(conv)
        result = tf.stack(convs, axis=2)
    return result


def k_max_pooling_2d(x, k, namespace="k_max_pooling", axis=1):
    if axis == 1:
        perm = [0, 3, 2, 1]
    elif axis == 2:
        perm = [0, 1, 3, 2]
    with tf.name_scope(namespace):
        x_t = tf.transpose(x, perm=perm)
        values = tf.nn.top_k(x_t, k, sorted=True).values
        out = tf.transpose(values, perm=perm)
    return out


def k_max_pooling_1d(x, k, namespace="k_max_pooling"):
    with tf.name_scope(namespace):
        x_t = tf.transpose(x, perm=[0, 2, 1])
        values = tf.nn.top_k(x_t, k, sorted=True).values
        out = tf.transpose(values, perm=[0, 2, 1])
    return out


def batch_euclid(x, y, name="euclid"):
    return batch_computation(x, y, name, square_euclid_distance)


def batch_manhattan(x, y, name="manhattan"):
    return batch_computation(x, y, name, manhattan_distance)


def square_euclid_distance(x, y):
    return tf.reduce_sum(tf.square(x - y), axis=1, keep_dims=True)


def euclid_distance(x, y):
    with tf.name_scope('euclidean_distance'):
        d = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(tf.subtract(x, y)), axis=1), 1e-6))
        return d


def manhattan_distance(x, y, keep_dims=True):
    return tf.reduce_sum(tf.abs(x - y), axis=1, keep_dims=keep_dims)


def cosine_distance(x, y):
    with tf.name_scope('cosine_distance'):
        x_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))
        y_norm = tf.sqrt(tf.reduce_sum(tf.square(y), axis=1))
        x_y = tf.reduce_sum(tf.multiply(x, y), axis=1)
        d = tf.divide(x_y, tf.maximum(tf.multiply(x_norm, y_norm), 1e-6))
        return d


def batch_computation(x, y, name, operation):
    with tf.name_scope(name):
        x_unstack = tf.unstack(x, axis=1)
        y_unstack = tf.unstack(y, axis=1)
        output_list = []
        for i in range(len(x_unstack)):
            row_list = []
            for j in range(len(y_unstack)):
                value = operation(x_unstack[i], y_unstack[j])
                row_list.append(value)
            row = tf.expand_dims(tf.concat(row_list, axis=1), -1)
            output_list.append(row)
        output = tf.concat(output_list, axis=2)
    return output
