# coding=utf-8

import os
import random
import numpy as np
import tensorflow as tf

import util.db_util as dbutil
import util.function_util as myutil
from flow.wordvector import load_word_embedding
from model.eval_single_class import EvalSingleClass
from model.train_model import TrainModel

# 反例是正例的5倍

model = None


def trainDataPrepare(sim_type="lda"):
    '''
    对于训练集，要选出候选反例
    :return:
    '''
    db = dbutil.get_mongodb_conn()
    cases_set = db.cases

    for line in cases_set.find({"flag": 2, "negftids": {"$exists": False}},
                               {"_id": 1, "ftids": 1, sim_type: 1},
                               no_cursor_timeout=True).batch_size(10):
        candi_statute = line[sim_type]

        # 构造反例
        neg_statutes = []
        ftidset = set(line["ftids"])
        max_neg_num = random.randint(1, 5) * len(line["ftids"])
        for ftid in candi_statute:
            if len(neg_statutes) == max_neg_num:
                break
            elif ftid not in ftidset:
                neg_statutes.append(ftid)

        cases_set.update(
            {"_id": line["_id"]},  # 更新条件
            {'$set': {"negftids": neg_statutes}},  # 更新内容
            upsert=False,  # 如果不存在update的记录，是否插入
            multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
        )


def trainCnn(type="cnn"):
    # 1:获取所有训练集id和验证集合id
    db = dbutil.get_mongodb_conn()
    cases_set = db.cases
    xml_names = []  # 训练集
    for line in cases_set.find({"flag": 2}, {"_id": 1}):
        xml_names.append(line["_id"])
    trial_xml_names = []  # 验证集
    for line in cases_set.find({"flag": 4}, {"_id": 1}):
        trial_xml_names.append(line["_id"])

    # 2:初始化模型
    conf = myutil.read_config("conf/fttj.conf")
    if type == "cnn":
        from model.cnn_model import CNNModel
        model = CNNModel(conf, load_word_embedding())   # 2.1 cnn模型
    elif type == "lstm":
        from model.lstm_model import LSTMModel
        model = LSTMModel(conf, load_word_embedding())
    elif type == "ann":
        from model.ann_model import ANNModel
        model = ANNModel(conf, load_word_embedding())
    eval_helper = EvalSingleClass()      # 2.2 评估模型
    train_helper = TrainModel(conf["learning_rate"])    # 2.3 获取train model
    train_op, global_step, train_summary_op = train_helper.get_train_model(model.loss)
    trial_summary_op = eval_helper.get_eval_summary()
    summary_writer = tf.summary.FileWriter("log/"+type, tf.get_default_graph())  # 2.4 可视化
    # 2.5 checkpoints
    checkpoint_prefix = os.path.abspath(os.path.join(os.path.curdir, "checkpoint/"+type, "model"))  # model是文件前缀，不是文件夹名
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=conf["num_checkpoints"])

    # 3: 开始训练
    run_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.InteractiveSession(config=run_config)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    for i in range(1, conf["num_epochs"] + 1):
        print("第%d轮：" % i)
        random.shuffle(xml_names)   # 每轮训练集洗牌
        # 3.1 训练集训练一轮
        for j in range(len(xml_names) // conf["batch_size"]):  # 一轮
            start = conf["batch_size"]*j
            end = start + conf["batch_size"]
            s1, s2, label = __transToData(xml_names[start:end])
            _, loss, step, train_summary, accuracy = sess.run(
                [train_op, model.loss, global_step, train_summary_op, model.accuracy],
                feed_dict={model.input_s1: s1,
                           model.input_s2: s2,
                           model.input_y: label,
                           model.dropout_keep_prob: conf["dropout_keep_prob"]})
            summary_writer.add_summary(train_summary, step)
            print("训练：", step, ", loss=", loss, ",accuracy=", accuracy)

        # 3.2 验证集验证一轮
        eval_helper.reset()
        # for j in range(len(trial_xml_names) // conf["batch_size"]):
        for j in range(50):     # 节省时间，只检验前500个
            start = conf["batch_size"] * j
            end = start + conf["batch_size"]
            s1, s2, label = __transToData(trial_xml_names[start:end])
            loss_batch, y = sess.run([model.loss, model.y],
                                           feed_dict={model.input_s1: s1,
                                                      model.input_s2: s2,
                                                      model.input_y: label,
                                                      model.dropout_keep_prob: 1.0})
            predict = np.greater_equal(y, 0.5)
            eval_helper.append(predict, label, loss_batch)
        accuracy, loss_value = eval_helper.get_accuracy()
        print("验证轮数", i, "：loss=", loss_value, ",accuracy=", accuracy)
        sess.run(eval_helper.get_assign_op())  # 要先提交值，后面的操作才会有改变
        trial_summaries = sess.run(trial_summary_op)
        summary_writer.add_summary(trial_summaries, i)

        # 4: 保存模型
        saver.save(sess, checkpoint_prefix, global_step=i)
        print("保存模型")


def __transToData(xml_names):
    db = dbutil.get_mongodb_conn()
    cases_set = db.cases
    statutes_set = db.statutes

    s1 = None
    s2 = None
    label = np.array([], dtype=np.int32)
    for xml_name in xml_names:
        case = cases_set.find_one({"_id": xml_name}, {"ygscid": 1, "ftids": 1, "negftids": 1})

        # 1：原告诉称重复正例＋反例次
        ygsc_array = np.repeat([case["ygscid"]], len(case["ftids"]) + len(case["negftids"]), axis=0)
        s1 = myutil.append_and_pad_2d_array(s1, ygsc_array)
        label = np.append(label, [1] * len(case["ftids"]) + [0] * len(case["negftids"]))
        for ftid in case["ftids"] + case["negftids"]:
            statute = statutes_set.find_one({"_id": ftid}, {"contentid": 1})
            s2 = myutil.append_and_pad_2d_array(s2, np.array([statute["contentid"]]))
    label = np.expand_dims(label, axis=-1)
    return s1, s2, label


def runCnn(ygscid, candi_statutes, type="cnn"):
    __loadModel(type)
    db = dbutil.get_mongodb_conn()
    statutes_set = db.statutes

    # 1: 得到输入数据
    s2 = None
    for ftid in candi_statutes:
        statute = statutes_set.find_one({"_id": ftid}, {"contentid": 1})
        s2 = myutil.append_and_pad_2d_array(s2, np.array([statute["contentid"]]))
    s1 = np.repeat([ygscid], s2.shape[0], axis=0)     # 根据s2复制s1
    label = np.repeat([[0]], s2.shape[0], axis=0)              # 根据s2生成label

    # 运行
    y = model.sess.run([model.predict_op],
                                feed_dict={model.input_s1: s1,
                                           model.input_s2: s2,
                                           model.input_y: label,
                                           model.dropout_keep_prob: 1.0})

    # 转换为法条index
    pred = np.greater_equal(y, 0.5).astype(np.int32).reshape(-1)
    recom_index = np.where(pred == 1)[0]
    recom_statutes = [candi_statutes[i] for i in recom_index]
    return recom_statutes, np.asarray(y).reshape(-1)


def __loadModel(type):
    global model
    if model is None:
        run_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.InteractiveSession(config=run_config)
        checkpoint_file = tf.train.latest_checkpoint("checkpoint/"+type)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        model = RunModel(sess)


class RunModel:
    def __init__(self, sess):
        self.sess = sess
        self.input_s1 = sess.graph.get_operation_by_name("input_s1").outputs[0]
        self.input_s2 = sess.graph.get_operation_by_name("input_s2").outputs[0]
        self.input_y = sess.graph.get_operation_by_name("input_y").outputs[0]
        self.dropout_keep_prob = sess.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        self.predict_op = sess.graph.get_operation_by_name("predict").outputs[0]