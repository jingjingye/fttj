# coding=utf-8

import util.db_util as dbutil
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.externals import joblib
from flow.cnn import runCnn

lr = None


def trainDataPrepare(nn_type="cnn", recom_num=35, sim_type="lda"):
    train_x = []
    train_y = []

    db = dbutil.get_mongodb_conn()
    cases_set = db.cases
    for line in cases_set.find({"flag": 2, sim_type: {"$exists": True}},
                               {"_id": 1, "ftids": 1, sim_type: 1, "ygscid": 1},
                               no_cursor_timeout=True).batch_size(20):
        # 1: 获取备选法条集
        statute_num = min(len(line[sim_type]), recom_num)
        candi_statute = line[sim_type][:statute_num]

        # 2: 构造输入
        lrinput = __get_lrinput(line["ygscid"], candi_statute, nn_type, recom_num)
        train_x.append(lrinput)

        # 3: 构造输出
        lroutput = np.zeros(statute_num, dtype=np.int32)
        for i, ftid in enumerate(candi_statute):
            if ftid in line["ftids"]:
                lroutput[i] = 1
        train_y.append(lroutput)

    # 4: 保存
    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    with open("checkpoint/lr_trainx.pk", "wb") as file:
        joblib.dump(train_x, file)
    with open("checkpoint/lr_trainy.pk", "wb") as file:
        joblib.dump(train_y, file)


def trainLR():
    with open("checkpoint/lr_trainx.pk", "rb") as file:
        train_x = joblib.load(file)
    with open("checkpoint/lr_trainy.pk", "rb") as file:
        train_y = joblib.load(file)

    lr = LogisticRegressionCV(class_weight="balanced", max_iter=1000, tol=1e-5)
    lr.fit(train_x, train_y)

    with open("checkpoint/lr.pk", "wb") as file:
        joblib.dump(lr, file)


def __load_object():
    global lr
    if lr is None:
        with open("checkpoint/lr.pk", "rb") as file:
            lr = joblib.load(file)


def __get_lrinput(ygscid, candi_statute, nn_type="cnn", recom_num=35):
    statute_num = len(candi_statute)
    # 1: 获取cnnscore
    _, cnn_score = runCnn(ygscid, candi_statute, nn_type)

    # 2: 拼接
    cnn_score = np.expand_dims(cnn_score, -1)
    refer_sort = np.eye(recom_num, dtype=np.float32)[:statute_num]
    lrinput = np.concatenate([cnn_score, refer_sort], axis=1)
    return lrinput


def runLR(ygscid, candi_statute, nn_type="cnn", recom_num=35):
    __load_object()
    # 1: 根据nn构造输入
    lrinput = __get_lrinput(ygscid, candi_statute, nn_type, recom_num)

    # 2: 利用lr预测
    lroutput = lr.predict(lrinput)

    # 3: 输出
    recom_index = np.where(lroutput == 1)[0]
    recom_statutes = [candi_statute[i] for i in recom_index]

    return recom_statutes