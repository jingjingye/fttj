# coding=utf-8
import util.db_util as dbutil
import util.function_util as myutil
from flow.similarCases import runSimilarCases, runCandiStatutes
from flow.cnn import runCnn
from flow.rules import runRules

# 通过simnsum的 k = 10


def run(flag, model_type, rules_on=True):
    '''
    测试（通过类案推荐）
    :param flag: 2是测试，3是验证
    :param model_type:  "simSum"、"cnn"
    :param rules_on:
    :return:
    '''
    # 存储的列名
    if rules_on:
        model_ft_name = model_type + "RulesFtids"
        model_precise_name = model_type + "RulesPrecise"
        model_recall_name = model_type + "RulesRecall"
    else:
        model_ft_name = model_type + "Ftids"
        model_precise_name = model_type + "Precise"
        model_recall_name = model_type + "Recall"

    precise_sum = 0
    recall_sum = 0
    case_num = 0
    ygscs = []
    xml_names = []

    db = dbutil.get_mongodb_conn()
    cases_set = db.cases

    # 获取测试案件
    for line in cases_set.find({"flag": flag}, {"_id": 1,"ygscWords2": 1}):
        ygscs.append(line["ygscWords2"])
        xml_names.append(line["_id"])

    # 1:获取相似案件
    simCases = runSimilarCases(ygscs)

    for i in range(len(xml_names)):
        # 2:根据相似案件获取候选法条
        sorted_candi_statutes = runCandiStatutes(simCases[i])

        # 3:进一步筛选法条
        recom_statute = []
        if model_type.startswith("simSum"):
            statute_num = min(len(sorted_candi_statutes), 10)
            recom_statute = [ftid for (ftid, ft_score) in sorted_candi_statutes[:statute_num]]
        elif model_type == "cnn":
            recom_statute = runCnn(xml_names[i], sorted_candi_statutes)

        # 4:伴随引用(可选)
        if rules_on:
            recom_statute = runRules(recom_statute)

        # 5:计算准确率
        ori_case = cases_set.find_one({"_id": xml_names[i]}, {"ftids": 1})
        right_statute_num = float(len(set(ori_case["ftids"]).intersection(set(recom_statute))))
        case_precise = right_statute_num/max(len(recom_statute), 0.1)   # 防止被除数为0
        case_recall = right_statute_num/len(ori_case["ftids"])
        # 5.1存入数据库记录
        cases_set.update(
            {"_id": xml_names[i]},  # 更新条件
            {'$set': {model_ft_name: recom_statute,
                      model_precise_name: case_precise,
                      model_recall_name: case_recall
                      }},  # 更新内容
            upsert=False,  # 如果不存在update的记录，是否插入
            multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
        )

        # 5.2计算到总的里
        precise_sum += case_precise
        recall_sum += case_recall
        case_num += 1

    logger = myutil.getLogger("test.log")
    precise = precise_sum/case_num
    recall = recall_sum/case_num
    logger.info("%s precise: %f" % (model_ft_name, precise))
    logger.info("%s recall: %f" % (model_ft_name, recall))


if __name__ == "__main__":
    run(3, "cnn", True)