# coding=utf-8
import tensorflow as tf
import os
import csv
import configparser
import logging
import numpy as np


# import scipy.stats as stats

def kl(y, input_y, name=None):
    kl = tf.reduce_mean(input_y * (tf.log(tf.clip_by_value(input_y, 1e-10, 1.0))
                                   - tf.log(tf.clip_by_value(y, 1e-10, 1.0))), name=name)
    return kl


def mse(y, input_y, name=None):
    mse = tf.reduce_mean(tf.square(y - input_y), name=name)
    return mse


def pearson(y, input_y, name=None):
    y_mean = tf.reduce_mean(y)
    input_y_mean = tf.reduce_mean(input_y)
    y_nor = y - y_mean
    input_y_nor = input_y - input_y_mean
    r_num = tf.reduce_sum(y_nor * input_y_nor)
    r_den = tf.sqrt(tf.reduce_sum(tf.square(y_nor)) * tf.reduce_sum(tf.square(input_y_nor)))
    pearsonr = tf.clip_by_value(r_num / r_den, -1.0, 1.0, name=name)

    # 计算方法2
    # mid1 = tf.reduce_mean(y * input_y) - tf.reduce_mean(y) * tf.reduce_mean(input_y)
    # mid2 = tf.sqrt(tf.reduce_mean(tf.square(y)) - tf.square(tf.reduce_mean(y))) * \
    #        tf.sqrt(tf.reduce_mean(tf.square(input_y)) - tf.square(tf.reduce_mean(input_y)))
    # pearsonr = tf.div(mid1, mid2, name=name)
    # 计算方法3 调用spicy
    # pearsonr, _ = tf.py_func(stats.pearsonr, [y, input_y], [tf.float32, tf.float32])
    return pearsonr


def spearman(y, input_y, vector_len, name=None):
    predictions_rank = tf.nn.top_k(y, k=vector_len, sorted=True, name="prediction_rank").indices
    real_rank = tf.nn.top_k(input_y, k=vector_len, sorted=True, name='real_rank').indices
    rank_diffs = predictions_rank - real_rank
    rank_diffs_squared_sum = tf.reduce_sum(tf.square(rank_diffs))
    numerator = tf.cast(6 * rank_diffs_squared_sum, dtype=tf.float32)
    divider = tf.cast(vector_len * vector_len * vector_len - vector_len, dtype=tf.float32)
    spearmanr = 1.0 - numerator / divider

    # 计算方法2 调用spicy
    # spearmanr, _ = tf.py_func(stats.spearmanr, [y, input_y], [tf.double, tf.double])
    return spearmanr


def get_file_num(filename):
    # 读文件
    total_num = len(os.listdir(filename))
    return total_num


def append_and_pad_2d_array(array, newlines):
    if array is None:
        return newlines
    else:
        diff = array.shape[1] - newlines.shape[1]
        if diff > 0:
            newlines = np.pad(newlines, ((0, 0), (0, diff)), mode="constant", constant_values=0)
        elif diff < 0:
            array = np.pad(array, ((0, 0), (0, -diff)), mode="constant", constant_values=0)
        array = np.concatenate([array, newlines], axis=0)
        return array


def record_result(filename, *columns):
    csvFile = open(filename, 'a', newline='', encoding="utf-8")  # 设置newline，否则两行之间会空一行
    writer = csv.writer(csvFile)
    data_num = len(columns[0])  # 一共多少行数据
    for i in range(data_num):
        writer.writerow([col[i] for col in columns])
    csvFile.close()


def read_config(filename):
    conf = {}
    cf = configparser.ConfigParser()
    cf.read(filename)
    if cf.has_section("string"):
        for key, value in cf.items("string"):
            conf[key] = value

    if cf.has_section("int"):
        for key, value in cf.items("int"):
            conf[key] = int(value)

    if cf.has_section("float"):
        for key, value in cf.items("float"):
            conf[key] = float(value)

    if cf.has_section("bool"):
        for key, value in cf.items("bool"):
            conf[key] = bool(int(value))

    return conf


def write_config(filename, section, option, value):
    cf = configparser.ConfigParser()
    cf.read(filename)
    cf.set(section, option, value)
    with open(filename, "w+") as f:
        cf.write(f)
    return


def create_conf_file(filename):
    if os.path.exists(filename):
        pos = filename.rfind('.')
        i = 1
        file = filename[:pos] + "_" + str(i) + filename[pos:]
        while os.path.exists(file):
            i += 1
            file = filename[:pos] + "_" + str(i) + filename[pos:]
    else:
        file = filename

    return file


def getLogger(fileName):
    logger = logging.getLogger('runtime')	#获取或创建名为runtime的logger
    logger.setLevel(logging.DEBUG)			#设置级别
    handler = logging.FileHandler("log/"+fileName, 'a')		#实例化handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)			#设置输出格式
    logger.addHandler(handler)
    return logger

