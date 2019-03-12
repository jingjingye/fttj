# coding=utf-8
from sklearn.neighbors import BallTree
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import util.db_util as dbutil
from sklearn.externals import joblib

# type：tfidf、lda、svd

# 超参数：
# k个最近邻：k=100
# 计算相似度的alpha：0.001

cv = None
xml_names = None
feature_model = None
ball_tree = None


def trainSimilarCases(firstTime=False, type="lda", vector_size=70):
    if firstTime:
        # 1：读ygsc及对应xml名
        ygscs = []
        xml_names = []
        labels = []
        db = dbutil.get_mongodb_conn()
        cases_set = db.cases
        for line in cases_set.find({"flag": 2}, {"_id": 1, "ygscWords2": 1, "label":1}, no_cursor_timeout=True).batch_size(20):
            ygscs.append(line["ygscWords2"])
            xml_names.append(line["_id"])
            labels.append(line["label"])

        ################## 2：转换为onehot   #####################
        cv = CountVectorizer(max_df=0.95, min_df=1, stop_words=None, token_pattern=r"(?u)\b\w+\b")
        one_hot_matrix = cv.fit_transform(ygscs).toarray()

        # 3：保存one-hot
        with open("checkpoint/cv.pk", "wb") as file:
            joblib.dump(cv, file)
        with open("checkpoint/onehot.pk", "wb") as file:
            joblib.dump(one_hot_matrix, file)
        with open("checkpoint/xml_names.pk", "wb") as file:
            joblib.dump(xml_names, file)
        with open("checkpoint/label.pk", "wb") as file:
            import numpy
            joblib.dump(numpy.asarray(labels), file)

        ################## 2'：转换为tfidf   ######################
        tfidf = TfidfVectorizer(max_df=0.95, min_df=1, stop_words=None, token_pattern=r"(?u)\b\w+\b")
        tf_idf_matrix = tfidf.fit_transform(ygscs).toarray()

        # 3'：保存tf-idf
        with open("checkpoint/tfidf.pk", "wb") as file:
            joblib.dump(tfidf, file)

        # 4：建balltree
        ball_tree = BallTree(tf_idf_matrix)
        with open("checkpoint/tfidf_ball_tree.pk", "wb") as file:
            joblib.dump(ball_tree, file)
    else:
        with open("checkpoint/onehot.pk", "rb") as file:
            one_hot_matrix = joblib.load(file)

    if type == "lda":
        # 0: 获取测试集
        test_ygscs = []
        db = dbutil.get_mongodb_conn()
        cases_set = db.cases
        for line in cases_set.find({"flag": 4}, {"ygscWords2": 1}, no_cursor_timeout=True).batch_size(20):
            test_ygscs.append(line["ygscWords2"])

        with open("checkpoint/cv.pk", "rb") as file:
            cv = joblib.load(file)
        test_matrix = cv.transform(test_ygscs).toarray()

        # 1：lda训练
        lda = LatentDirichletAllocation(n_components=vector_size)
        lda_matrix = lda.fit_transform(one_hot_matrix)

        # 2: 评估
        print("lda" + str(vector_size) + "train困惑度：" + str(lda.perplexity(one_hot_matrix)))  # 查看困惑度
        print("lda" + str(vector_size) + "test困惑度：" + str(lda.perplexity(test_matrix)))  # 查看困惑度

        # 3：保存
        with open("checkpoint/lda.pk", "wb") as file:
            joblib.dump(lda, file)
        with open("checkpoint/feature_matric.pk", "wb") as file:
            joblib.dump(lda_matrix, file)
        # 4：建balltree
        ball_tree = BallTree(lda_matrix)
        with open("checkpoint/lda_ball_tree.pk", "wb") as file:
            joblib.dump(ball_tree, file)
    elif type == "svd":
        # 1：svd训练
        svd = TruncatedSVD(n_components=vector_size)        # 迭代次数
        svd_matrix = svd.fit_transform(one_hot_matrix)

        # 2：保存
        with open("checkpoint/svd.pk", "wb") as file:
            joblib.dump(svd, file)

        # 3：建balltree
        ball_tree = BallTree(svd_matrix)
        with open("checkpoint/svd_ball_tree.pk", "wb") as file:
            joblib.dump(ball_tree, file)


def __load_object(type="lda"):
    global cv, xml_names, feature_model, ball_tree
    if cv is None or xml_names is None or feature_model is None or ball_tree is None:
        with open("checkpoint/cv.pk", "rb") as file:
            cv = joblib.load(file)
        with open("checkpoint/xml_names.pk", "rb") as file:
            xml_names = joblib.load(file)

        if type == "lda":
            with open("checkpoint/lda.pk", "rb") as file:
                feature_model = joblib.load(file)
            with open("checkpoint/lda_ball_tree.pk", "rb") as file:
                ball_tree = joblib.load(file)
        elif type == "svd":
            with open("checkpoint/svd.pk", "rb") as file:
                feature_model = joblib.load(file)
            with open("checkpoint/svd_ball_tree.pk", "rb") as file:
                ball_tree = joblib.load(file)
        elif type == "tfidf":
            with open("checkpoint/tfidf.pk", "rb") as file:
                feature_model = joblib.load(file)
            with open("checkpoint/tfidf_ball_tree.pk", "rb") as file:
                ball_tree = joblib.load(file)

def __compute_sim_score(dis):
    '''
    由距离转换为相似度
    :param dis:
    :return:
    '''
    score = 1/(dis+0.001)
    return score


def runSimilarCases(strArray, type, k_num=30):
    '''

    :param strArray:
    :return: 对每个case，是(xml_name, 相似度)的列表。整个是所有case的二维列表
    '''
    __load_object(type)
    rlt = []

    # 1：获取特征向量
    case_vector = get_case_vector(strArray, type)

    # 2：寻找k近邻
    dis, ind = ball_tree.query(case_vector, k=k_num)  # k个近邻
    for i in range(len(ind)):   # 第i个str
        sim_cases = []
        for j in range(len(ind[0])):    # 找到的第j个相似的案件
            sim_cases.append((xml_names[ind[i][j]], __compute_sim_score(dis[i][j])))
        rlt.append(sim_cases)
    return rlt


def get_case_vector(strArray, type):
    __load_object(type)

    if type == "tfidf":
        case_vector = feature_model.transform(strArray).toarray()
    elif type == "lda" or type == "svd":
        one_hot = cv.transform(strArray).toarray()   # onehot
        case_vector = feature_model.transform(one_hot)

    return case_vector



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