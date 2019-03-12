# coding=utf-8
import numpy as np
from model.eval_model import EvalModel


class EvalMultiClass(EvalModel):

    def get_accuracy(self):
        predict_flat = np.concatenate(self.predict, 0)
        score_flat = np.concatenate(self.score, 0)
        predict_sum = np.maximum(np.sum(predict_flat, axis=1).astype(np.float32), 0.001)
        score_sum = np.sum(score_flat, axis=1).astype(np.float32)
        predict_right = np.sum(np.logical_and(predict_flat, score_flat), axis=1).astype(np.float32)
        self.accuracy = np.mean(predict_right/predict_sum)
        self.recall = np.mean(predict_right/score_sum)
        self.loss = np.mean(self.loss_set)

        return self.accuracy, self.recall, self.loss

    def get_assign_op(self):
        import tensorflow as tf
        return tf.group(tf.assign(self.loss_op, self.loss), tf.assign(self.acc_op, self.accuracy),
                        tf.assign(self.recall_op, self.recall))

    def get_eval_summary(self):
        import tensorflow as tf
        with tf.variable_scope("eval"):
            self.loss_op = tf.get_variable("loss", shape=[], initializer=tf.constant_initializer(0.0))
            self.acc_op = tf.get_variable("acc", shape=[], initializer=tf.constant_initializer(0.0))
            self.recall_op = tf.get_variable("recall", shape=[], initializer=tf.constant_initializer(0.0))

        # 可视化
        acc_summary = tf.summary.scalar("trial/accuracy/acc", self.acc_op)
        recall_summary = tf.summary.scalar("trial/accuracy/recall", self.recall_op)
        loss_summary = tf.summary.scalar("trial/accuracy/loss", self.loss_op)
        merged = tf.summary.merge([acc_summary, recall_summary, loss_summary])
        return merged
