# coding=utf-8
import tensorflow as tf
import util.model_util as model_util


class LSTMModel(object):

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

        with tf.variable_scope("lstm"):
            basic_cell = tf.nn.rnn_cell.BasicLSTMCell(word_embedding.shape[1])
            s1_out = get_rnn_output(s1, basic_cell)
            s2_out = get_rnn_output(s2, basic_cell)
            fc_input = tf.concat([s1_out, s2_out], 1)

        total_num = word_embedding.shape[1] * 2
        # 全链接层
        fc_drop = tf.nn.dropout(fc_input, self.dropout_keep_prob)
        output = model_util.get_full_connect_layer(fc_drop, total_num, 1, "output", regularization=True)

        labels = tf.cast(self.input_y, tf.float32)
        self.y = tf.nn.sigmoid(output, name="predict")
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=labels)
            self.loss = tf.reduce_mean(losses, name="cross_entropy")

        with tf.name_scope("accuracy"):
            predict = tf.cast(tf.round(self.y), dtype=tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, self.input_y), tf.float32))


def get_rnn_output(x, basic_cell):
    outputs, states = tf.nn.dynamic_rnn(basic_cell, x, dtype=tf.float32)
    lstm_out = outputs[:, -1, :]
    return lstm_out