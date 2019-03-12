# coding=utf-8

import util.function_util as myutil
from flow.wordvector import load_word_embedding
from annoy import AnnoyIndex
from sklearn.neighbors import BallTree, KDTree
from sklearn.externals import joblib
import numpy
import time


def top_k_train():
    logger = myutil.getLogger("try.log")

    emd = load_word_embedding()
    logger.info("vector size: %d" % len(emd))

    annoy_model = AnnoyIndex(300)
    for (i, vec) in enumerate(emd):
        annoy_model.add_item(i, vec)
    annoy_model.build(50)   # 建20棵树,树越大越精确
    annoy_model.save('checkpoint/annoy.pk')

    ball_tree = BallTree(emd)
    with open("checkpoint/ball_tree.pk", "wb") as file:
        joblib.dump(ball_tree, file)

    kd_tree = KDTree(emd)
    with open("checkpoint/kd_tree.pk", "wb") as file:
        joblib.dump(kd_tree, file)


def top_k_test(type="annoy"):
    logger = myutil.getLogger("try.log")

    if type == "annoy":
        model = AnnoyIndex(300)
        model.load('checkpoint/annoy.pk')
    elif type == "kd_tree":
        with open("checkpoint/kd_tree.pk", "rb") as file:
            model = joblib.load(file)
    elif type == "ball_tree":
        with open("checkpoint/ball_tree.pk", "rb") as file:
            model = joblib.load(file)

    max_num = 0.0
    min_num = 1000.0
    sum_num = 0.0
    count = 0
    for i in range(100000):
        vec = numpy.random.uniform(-1, 1, size=300)
        start = time.time()
        if type == "annoy":
            # 通过第几个item查询：get_nns_by_item  通过向量查询：get_nns_by_vector
            words, dis = model.get_nns_by_vector(vec, 100, include_distances=True)
            # for id in words:
            #     print(id)
        else:
            dis, ind = model.query([vec], k=100)
            # for j in range(len(ind[0])):
            #     print(ind[0][j], dis[0][j])
        stop = time.time()

        # 更新
        run_time = float(stop-start)
        sum_num += run_time
        count += 1
        if run_time > max_num:
            max_num = run_time
        if run_time < min_num:
            min_num = run_time

    logger.info("%s, max: %f, min: %f, avg: %f, count: %f" %
                    (type, max_num, min_num, (sum_num/count), count))


if __name__ == "__main__":
    # top_k_train()
    for type in ["annoy", "kd_tree", "ball_tree"]:
        top_k_test(type)