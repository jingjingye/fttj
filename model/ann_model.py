# coding=utf-8
import tensorflow as tf
import util.model_util as model_util


class ANNModel(object):

    def __init__(self, conf, word_embedding):
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.input_y = tf.placeholder(tf.int32, shape=[None, 1], name="input_y")
        self.input_s1 = tf.placeholder(tf.int32, shape=[None, None], name="input_s1")
        self.input_s2 = tf.placeholder(tf.int32, shape=[None, None], name="input_s2")

        with tf.variable_scope("embedding"):
            word_embedding_W = tf.get_variable(name='word_embedding', shape=word_embedding.shape,
                                               dtype=tf.float32,
                                               initializer=tf.constant_initializer(word_embedding),
                                               trainable=False)
            s1 = tf.nn.embedding_lookup(word_embedding_W, self.input_s1)
            s2 = tf.nn.embedding_lookup(word_embedding_W, self.input_s2)
            s1_average = tf.reduce_sum(s1, 1)/tf.cast(tf.count_nonzero(self.input_s1, axis=1, keepdims=True), dtype=tf.float32)
            s2_average = tf.reduce_sum(s2, 1)/tf.cast(tf.count_nonzero(self.input_s2, axis=1, keepdims=True), dtype=tf.float32)

        s1_hidden = model_util.get_full_connect_layer(s1_average, word_embedding.shape[1], 2400, "hidden_one",
                                                      regularization=True)
        s2_hidden = model_util.get_full_connect_layer(s2_average, word_embedding.shape[1], 2400, "hidden_one",
                                                      reuse=True, regularization=False)

        with tf.variable_scope("concat"):
            s1_dot_s2 = s1_hidden * s2_hidden
            s1_abs_s2 = tf.abs(s1_hidden - s2_hidden)
            s = tf.concat([s1_dot_s2, s1_abs_s2], 1)

        hidden = model_util.get_full_connect_layer(s, 2400 * 2, 50, "hidden_two",
                                                   regularization=True)

        fc_drop = tf.nn.dropout(hidden, self.dropout_keep_prob)
        output = model_util.get_full_connect_layer(fc_drop, 50, 1, "output", regularization=True)

        labels = tf.cast(self.input_y, tf.float32)
        self.y = tf.nn.sigmoid(output, name="predict")
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=labels)
            self.loss = tf.reduce_mean(losses, name="cross_entropy")

        with tf.name_scope("accuracy"):
            predict = tf.cast(tf.round(self.y), dtype=tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, self.input_y), tf.float32))

