# coding=utf-8

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
from sklearn.externals import joblib

from flow.similarCases import get_case_vector
import util.db_util as dbutil
import util.function_util as myutil

model = None


def prepareLabels(flag=2):
    '''
    准备训练集或验证集的label
    :param flag:
    :return:
    '''
    logger = myutil.getLogger("label.log")
    statute_dict = {}
    statute_index = 0
    db = dbutil.get_mongodb_conn()
    cases_set = db.cases

    if flag == 2:
        statutes_set = db.statutes
        statute_num = statutes_set.count({"sampleTrainCount": {"$exists": True}})
    else:
        with open("checkpoint/statute_dict.pk", "rb") as file:
            statute_dict = joblib.load(file)
        statute_num = len(statute_dict)

    for line in cases_set.find({"flag": flag}, {"ftids": 1}, no_cursor_timeout=True).batch_size(20):
        logger.info(line["_id"])
        label = [0 for i in range(statute_num)]
        legal = True
        for ftid in line["ftids"]:
            if ftid in statute_dict:
                label[statute_dict[ftid]] = 1  # 直接赋值为1
            else:
                if flag == 2:
                    statute_dict[ftid] = statute_index  # 加入dict里面没有的
                    label[statute_index] = 1  # 赋值为1
                    statute_index += 1  # 更新计数
                else:
                    logger.error("出现不在训练集的法条：%s" % line["_id"])
                    legal = False
                    break

        if legal:
            cases_set.update(
                {"_id": line["_id"]},  # 更新条件
                {'$set': {"label": label}},  # 更新内容
                upsert=False,  # 如果不存在update的记录，是否插入
                multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
            )

    # 保存dict
    if flag == 2:
        with open("checkpoint/statute_dict.pk", "wb") as file:
            joblib.dump(statute_dict, file)


def trainMulti(type="decision_tree"):
    with open("checkpoint/feature_matric.pk", "rb") as file:
        x = joblib.load(file)
    with open("checkpoint/label.pk", "rb") as file:
        y = joblib.load(file)

    if type == "decision_tree":
        model = DecisionTreeClassifier()
    elif type == "random_forest":
        model = RandomForestClassifier()
    elif type == "svm":
        model = BinaryRelevance(SVC())
    model.fit(x, y)

    with open("checkpoint/" + type + ".pk", "wb") as file:
        joblib.dump(model, file)


def runMulti(strArray, type="decision_tree"):
    __load_model(type)
    case_vector = get_case_vector(strArray, "lda")
    recom = model.predict(case_vector)
    if type == "svm":
        recom = recom.asarray()
    return recom


def __load_model(type):
    global model
    if model is None:
        with open("checkpoint/" + type + ".pk", "rb") as file:
            model = joblib.load(file)