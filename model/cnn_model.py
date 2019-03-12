# coding=utf-8
import tensorflow as tf
import util.model_util as model_util


class CNNModel(object):

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

        with tf.variable_scope("feature"):
            weight = tf.get_variable("weights", shape=[conf["embedding_size"], conf["embedding_size"]],
                                initializer=tf.contrib.layers.xavier_initializer())
            w_tmp = tf.tile(weight, [tf.shape(s1)[0], 1])
            weight_reshape = tf.reshape(w_tmp, [tf.shape(s1)[0], tf.shape(weight)[0], tf.shape(weight)[1]])

            dot_s1 = tf.matmul(s1, weight_reshape)
            dot = tf.matmul(dot_s1, tf.transpose(s2, perm=[0, 2, 1]), name="dot")

            # manhattan = model_util.batch_manhattan(s1, s2, name="manhattan")
            # euclid = model_util.batch_euclid(s1, s2, name="euclid")
            # x = combine_feature(dot, manhattan, euclid)
            x = tf.expand_dims(dot, -1)

        pooled_outputs = []
        filter_sizes = list(map(int, conf["filter_sizes"].split(",")))
        for filter_size in filter_sizes:
            conv = model_util.get_conv2d_layer(x, filter_size, filter_size, 1, conf["num_filters"],
                                               "conv-maxpool-%s" % filter_size, padding="SAME")
            pool_max = tf.reduce_max(conv, axis=2, keep_dims=True, name="pool_max-%s" % filter_size)
            pooled_outputs.append(pool_max)
        conv_concat = tf.concat(pooled_outputs, 2, name="pool_concat")
        conv2 = model_util.get_conv2d_layer(conv_concat, 1, len(filter_sizes), conf["num_filters"], conf["num_filters_2"], "conv2", padding="VALID")
        pooled_concat = model_util.k_max_pooling_2d(conv2, conf["pool_k"], namespace="pool_max_k")

        total_num = conf["num_filters_2"] * conf["pool_k"]
        # 全链接层
        fc_input = tf.reshape(pooled_concat, [-1, total_num])
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


# def combine_feature(dot_matrix, manhattan_matrix, euclid_matrix):
#     # mask_matrix = tf.matmul(mask1, tf.transpose(mask2, perm=[0, 2, 1]))
#     with tf.variable_scope("feature"):
#         dot = normalize_feature(dot_matrix, 0.3, "dot")
#         manhattan = normalize_feature(manhattan_matrix, 0.1, "manhattan")
#         euclid = normalize_feature(euclid_matrix, 0.2, "euclid")
#         feature = tf.concat([dot, manhattan, euclid], axis=3)
#     return feature
#
#
# def normalize_feature(matrix, w_initial, namespace):
#     with tf.variable_scope(namespace):
#         W = tf.get_variable("W", shape=[], initializer=tf.constant_initializer(w_initial),
#                             dtype=tf.float32, trainable=False)
#         out = tf.expand_dims(W * matrix, -1)
#     return out
