# coding=utf-8
import util.db_util as dbutil
import util.function_util as myutil

logger = None


def genSim(flag=4, type="lda", sim_case_num=30):
    '''
    生成备选法条集（通过类案推荐）
    :param flag: 2是训练，4是测试
    :param type:  "lda"、"svd"、"tfidf"
    :param sim_case_num:  类案数字
    :return:
    '''
    from flow.similarCases import runSimilarCases, runCandiStatutes

    db = dbutil.get_mongodb_conn()
    cases_set = db.cases

    for line in cases_set.find({"flag": flag, type: {"$exists": False}},
                               {"_id": 1, "ygscWords2": 1}, no_cursor_timeout=True).batch_size(20):
        # 1:获取相似案件
        simCases = runSimilarCases([line["ygscWords2"]], type, sim_case_num)

        # 2:根据相似案件获取候选法条
        sorted_candi_statutes = runCandiStatutes(simCases[0])

        # 3:转换成列表
        statute_num = min(len(sorted_candi_statutes), 500)  # 最多不能超过500个
        recom_statute = [ftid for (ftid, ft_score) in sorted_candi_statutes[:statute_num]]

        # 4：存储
        cases_set.update(
            {"_id": line["_id"]},  # 更新条件
            {'$set': {type: recom_statute}},  # 更新内容
            upsert=False,  # 如果不存在update的记录，是否插入
            multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
        )


def clearSim(type="lda"):
    '''
    清空生成相似的记录
    :param type:
    :return:
    '''
    db = dbutil.get_mongodb_conn()
    cases_set = db.cases
    cases_set.update(
        {"flag": 4},  # 更新条件
        {'$unset': {type: 1}},  # 更新内容
        upsert=False,  # 如果不存在update的记录，是否插入
        multi=True,  # 可选，mongodb 默认是false,只更新找到的第一条记录
    )


def evaluateSim(recom_num, type="lda"):     # 只有测试集需要跑这个
    # 存储入数据库的列名
    model_precise_name = type + str(recom_num) + "Precise"
    model_recall_name = type + str(recom_num) + "Recall"
    # sum
    precise_sum = 0
    recall_sum = 0
    case_num = 0

    db = dbutil.get_mongodb_conn()
    cases_set = db.cases
    for line in cases_set.find({"flag": 4, type: {"$exists": True}},
                               {"_id": 1, "ftids": 1, type: 1}, no_cursor_timeout=True).batch_size(20):
        # 1: 获取推荐列表
        statute_num = min(len(line[type]), recom_num)
        recom_statute = line[type][:statute_num]

        # 2: 计算精度、召回
        case_precise, case_recall = __get_precise_and_recall(recom_statute, line["ftids"])

        # 3：加到总的里
        precise_sum += case_precise
        recall_sum += case_recall
        case_num += 1

        # 4: 存入数据库记录
        cases_set.update(
            {"_id": line["_id"]},  # 更新条件
            {'$set': {model_precise_name: case_precise,
                      model_recall_name: case_recall
                      }},  # 更新内容
            upsert=False,  # 如果不存在update的记录，是否插入
            multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
        )

    # 计算总的精度、召回
    __print_total_precise_and_recall(precise_sum, recall_sum, case_num, type + str(recom_num))


def testCnn(model="cnn", recom_num=100, sim_type="lda"):
    '''
    测试文本相关性
    :return:
    '''
    from flow.cnn import runCnn

    # 存储入数据库的列名
    model_precise_name = model + "Precise"
    model_recall_name = model + "Recall"
    # sum
    precise_sum = 0
    recall_sum = 0
    case_num = 0

    db = dbutil.get_mongodb_conn()
    cases_set = db.cases
    for line in cases_set.find({"flag": 4, sim_type: {"$exists": True}},
                               {"_id": 1, "ftids": 1, sim_type: 1, "ygscid": 1},
                               no_cursor_timeout=True).batch_size(20):
        # 1: 获取备选法条集
        statute_num = min(len(line[sim_type]), recom_num)
        candi_statute = line[sim_type][:statute_num]

        # 2: 送到模型中运行
        recom_statute = runCnn(line["ygscid"], candi_statute, model)

        # 2: 计算精度、召回
        case_precise, case_recall = __get_precise_and_recall(recom_statute, line["ftids"])

        # 3：加到总的里
        precise_sum += case_precise
        recall_sum += case_recall
        case_num += 1

        # 4: 存入数据库记录
        cases_set.update(
            {"_id": line["_id"]},  # 更新条件
            {'$set': {model: recom_statute,
                      model_precise_name: case_precise,
                      model_recall_name: case_recall
                      }},  # 更新内容
            upsert=False,  # 如果不存在update的记录，是否插入
            multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
        )

        # 计算总的精度、召回
    __print_total_precise_and_recall(precise_sum, recall_sum, case_num, model)


def testRules(text_model="cnn"):
    from flow.rules import runRules

    precise_sum = 0
    recall_sum = 0
    case_num = 0

    db = dbutil.get_mongodb_conn()
    cases_set = db.cases
    for line in cases_set.find({"flag": 4},
                               {"_id": 1, "ftids": 1, text_model: 1}, no_cursor_timeout=True).batch_size(20):
        recom_statute = runRules(line[text_model])

        # 2: 计算精度、召回
        case_precise, case_recall = __get_precise_and_recall(recom_statute, line["ftids"])

        # 3：加到总的里
        precise_sum += case_precise
        recall_sum += case_recall
        case_num += 1

        # 4: 存入数据库记录
        cases_set.update(
            {"_id": line["_id"]},  # 更新条件
            {'$set': {"rules": recom_statute,
                      "rulesPrecise": case_precise,
                      "rulesRecall": case_recall
                      }},  # 更新内容
            upsert=False,  # 如果不存在update的记录，是否插入
            multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
        )

    # 计算总的精度、召回
    __print_total_precise_and_recall(precise_sum, recall_sum, case_num, "rules")


def testMulti(multi_model="svm"):
    from flow.multiLabel import runMulti
    import numpy as np

    precise_sum = 0
    recall_sum = 0
    case_num = 0

    db = dbutil.get_mongodb_conn()
    cases_set = db.cases
    for line in cases_set.find({"flag": 4}, {"_id": 1, "ygscWords2": 1, "label": 1},
                               no_cursor_timeout=True).batch_size(20):
        recom_statute = runMulti([line["ygscWords2"]], multi_model)

        # 2: 计算精度、召回
        predict_sum = np.maximum(np.sum(recom_statute, axis=1).astype(np.float32), 0.001)
        score_sum = np.sum(line["label"], axis=1).astype(np.float32)
        predict_right = np.sum(np.logical_and(recom_statute, line["label"]), axis=1).astype(np.float32)
        case_precise = np.mean(predict_right / predict_sum)
        case_recall = np.mean(predict_right / score_sum)

        # 3：加到总的里
        precise_sum += case_precise
        recall_sum += case_recall
        case_num += 1

        # 4: 存入数据库记录
        # cases_set.update(
        #     {"_id": line["_id"]},  # 更新条件
        #     {'$set': {multi_model: recom_statute,
        #               multi_model + "Precise": case_precise,
        #               multi_model + "Recall": case_recall
        #               }},  # 更新内容
        #     upsert=False,  # 如果不存在update的记录，是否插入
        #     multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
        # )

    # 计算总的精度、召回
    __print_total_precise_and_recall(precise_sum, recall_sum, case_num, multi_model)


def __get_precise_and_recall(y, target):
    right_statute_num = float(len(set(target).intersection(set(y))))
    precise = right_statute_num / max(len(y), 0.1)  # 防止被除数为0
    recall = right_statute_num / len(target)
    return precise, recall


def __load_logger(logfile="test.log"):
    global logger
    if logger is None:
        logger = myutil.getLogger(logfile)


def __print_total_precise_and_recall(precise_sum, recall_sum, case_num, model_type, logfile="test.log"):
    __load_logger(logfile)
    precise = precise_sum / case_num
    recall = recall_sum / case_num
    logger.info("%s precise: %f" % (model_type, precise))
    logger.info("%s recall: %f" % (model_type, recall))


def fine_sim_param(sim_type="lda"):
    __load_logger()
    for sim_num in range(1, 50, 2):
        logger.info("#############%s,m=%d###########" % (sim_type, sim_num))
        genSim(flag=4, type=sim_type)
        for top_k in range(5, 70, 5):
            evaluateSim(top_k, type=sim_type)
        clearSim(sim_type)


def fine_rule_param():
    from flow.rules import trainRules
    import numpy as np

    __load_logger()
    db = dbutil.get_mongodb_conn()
    rules_set = db.rules

    for minsup in range(5, 31, 5):
        for minconf in np.arange(0.6, 1, 0.1):
            logger.info("#############rules: minsup=%d, minconf=%f ###########" % (minsup, minconf))
            rules_set.drop()
            trainRules(minsup, minconf)
            testRules(text_model="cnn")


if __name__ == "__main__":
    # fine_sim_param("tfidf")

    # genSim(flag=4, type="tfidf")
    # evaluateSim(30, type="lda")
    # testCnn(model="ann", recom_num=100, sim_type="tfidf")

    fine_rule_param()
    # testRules(text_model="cnn")

    # testMulti("svm")