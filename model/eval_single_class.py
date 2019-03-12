# coding=utf-8
import numpy as np
from model.eval_model import EvalModel


class EvalSingleClass(EvalModel):

    def get_accuracy(self):
        predict_flat = np.concatenate(self.predict, 0)
        score_flat = np.concatenate(self.score, 0)
        self.accuracy = np.mean(np.equal(predict_flat, score_flat).astype(np.float32))
        self.loss = np.mean(self.loss_set)

        return self.accuracy, self.loss

    def get_assign_op(self):
        import tensorflow as tf
        return tf.group(tf.assign(self.loss_op, self.loss), tf.assign(self.acc_op, self.accuracy))

    def get_eval_summary(self):
        import tensorflow as tf
        with tf.variable_scope("eval"):
            self.loss_op = tf.get_variable("loss", shape=[], initializer=tf.constant_initializer(0.0))
            self.acc_op = tf.get_variable("acc", shape=[], initializer=tf.constant_initializer(0.0))

        # 可视化
        acc_summary = tf.summary.scalar("trial/accuracy/acc", self.acc_op)
        loss_summary = tf.summary.scalar("trial/accuracy/loss", self.loss_op)
        merged = tf.summary.merge([acc_summary, loss_summary])
        return merged
