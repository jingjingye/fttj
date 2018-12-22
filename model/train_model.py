# coding=utf-8
import tensorflow as tf


class TrainModel(object):
    def __init__(self, learning_rate):
        self.global_step = tf.get_variable(name="global_step", shape=[], initializer=tf.constant_initializer(0),
                                           trainable=False)  # 计数器
        self.optimizer = tf.train.AdamOptimizer(learning_rate)

    def get_train_model(self, loss):
        # 优化
        grads_and_vars = self.optimizer.compute_gradients(loss)
        grad_summary = TrainModel.__grad_summary(grads_and_vars)
        train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # 可视化
        loss_summary = tf.summary.scalar("train/loss_summary", loss)
        merged = tf.summary.merge([grad_summary, loss_summary])
        return train_op, self.global_step, merged

    @staticmethod
    def __grad_summary(grads_and_vars):
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        return grad_summaries_merged
