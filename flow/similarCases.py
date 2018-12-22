# coding=utf-8
from sklearn.neighbors import BallTree
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import util.db_util as dbutil
import pickle

# 超参数：
# k个最近邻：k=100
# 计算相似度的alpha：0.001

cv = None
xml_names = None
lda = None
ball_tree = None


def trainSimilarCases():
    # 读ygsc及对应xml名
    ygscs = []
    xml_names = []
    db = dbutil.get_mongodb_conn()
    cases_set = db.cases
    for line in cases_set.find({"flag": 2}):
        ygscs.append(line["ygscWords2"])
        xml_names.append(line["_id"])

    # 转换为onehot
    cv = CountVectorizer(max_df=0.95, min_df=1, stop_words=None)
    one_hot_matrix = cv.fit_transform(ygscs).toarray()

    # lda训练
    lda = LatentDirichletAllocation(n_topics=200,
                                    max_iter=50,
                                    max_doc_update_iter=100,
                                    learning_method='online',
                                    learning_decay=0.7,
                                    learning_offset=10.0,
                                    batch_size=128)
    lda_matrix = lda.fit_transform(one_hot_matrix)

    # 建balltree
    ball_tree = BallTree(lda_matrix)

    # 保存
    with open("checkpoint/onehot.pk", "wb") as file:
        pickle.dump(cv, file)
    with open("checkpoint/xml_names.pk", "wb") as file:
        pickle.dump(xml_names, file)
    with open("checkpoint/lda.pk", "wb") as file:
        pickle.dump(lda, file)
    with open("checkpoint/ball_tree.pk", "wb") as file:
        pickle.dump(ball_tree, file)


def __load_object():
    global cv, xml_names, lda, ball_tree
    if cv is None or xml_names is None or lda is None or ball_tree is None:
        with open("checkpoint/onehot.pk", "rb") as file:
            cv = pickle.load(file)
        with open("checkpoint/xml_names.pk", "rb") as file:
            xml_names = pickle.load(file)
        with open("checkpoint/lda.pk", "rb") as file:
            lda = pickle.load(file)
        with open("checkpoint/ball_tree.pk", "rb") as file:
            ball_tree = pickle.load(file)


def __compute_sim_score(dis):
    '''
    由距离转换为相似度
    :param dis:
    :return:
    '''
    score = 1/(dis+0.001)
    return score


def runSimilarCases(strArray):
    '''

    :param strArray:
    :return: 对每个case，是(xml_name, 相似度)的列表。整个是所有case的二维列表
    '''
    __load_object()
    rlt = []

    one_hot = cv.transform(strArray).toarray()   # onehot
    lda_vector = lda.transform(one_hot)     # lda
    dis, ind = ball_tree.query(lda_vector, k=100)  # k个近邻
    for i in range(len(ind)):   # 第i个str
        sim_cases = []
        for j in range(len(ind[0])):    # 找到的第j个相似的案件
            sim_cases.append((xml_names[ind[i][j]], __compute_sim_score(dis[i][j])))
        rlt.append(sim_cases)
    return rlt


def runCandiStatutes(simCases):
    '''
    获取关联法条
    :param simCases:
    :return: (ftid, score)的列表
    '''
    db = dbutil.get_mongodb_conn()
    cases_set = db.cases
    candi_statutes = {}
    for (sim_id, sim_score) in simCases:
        sim_case = cases_set.find_one({"_id": sim_id}, {"ftids": 1})
        for ftid in sim_case["ftids"]:
            candi_statutes[ftid] = candi_statutes.get(ftid, 0) + sim_score
    # 3:将候选法条按分数倒序
    sorted_candi_statutes = sorted(candi_statutes.items(), key=lambda x: x[1], reverse=True)
    return sorted_candi_statutes