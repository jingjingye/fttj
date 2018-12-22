# coding=utf-8
import abc


class EvalModel(metaclass=abc.ABCMeta):
    def __init__(self):
        self.reset()

    def reset(self):
        self.predict = []
        self.score = []
        self.loss_set = []

    def append(self, predict_batch, score_batch, loss_batch):
        self.predict.append(predict_batch)
        self.score.append(score_batch)
        self.loss_set.append(loss_batch)

    @abc.abstractclassmethod
    def get_accuracy(self):
        return

    @abc.abstractclassmethod
    def get_assign_op(self):
        return

    @abc.abstractclassmethod
    def get_eval_summary(self):
        return
