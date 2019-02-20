# coding=utf-8

import jieba.posseg as psg
import util.db_util as dbutil
import util.function_util as myutil
import random
import math
import re

# 超参数：
# 训练测试验证占比：451，
# 最短的ygsc长度(两处)：20，
# 停用词最小的熵2.0，停用词词频>=200000、<=50，
# 不重复的窗口大小5


def fenci(content):
    # 去除英文
    pattern = re.compile(r'[a-zA-Z某]')  # 1:不是英文
    content = re.sub(pattern, '', content)

    words = psg.cut(content)
    words_list = []
    for (w, flag) in words:
        if not (len(w) == 1 or w.isspace() or w.isdigit() or flag == 'w' or flag == 'x'):  # 2:不是单字，不是空白，不是数字，不是符号
            words_list.append(w)
    return words_list


def compute_words_entropy():
    db = dbutil.get_mongodb_conn()
    words_set = db.words
    for line in words_set.find():
        total = float(line["totalCount"])
        entropy = 0.0
        for (aydm, ay_count) in line["ayCount"].items():
            prop = ay_count/total
            entropy -= prop * math.log(prop)
        words_set.update(
            {"_id": line["_id"]},
            {'$set': {"entropy": entropy}},
            upsert=False,  # 如果不存在update的记录，是否插入
            multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
        )


def print_stopwords():
    db = dbutil.get_mongodb_conn()
    words_set = db.words
    for line in words_set.find({"entropy": {"$gte": 2.4}, "totalCount": {"$gte": 60000}},
                                    {"_id": 1}, no_cursor_timeout=True).batch_size(10):
        print(line["_id"], end=" ")


def __not_stopwords(word_db):
    if (word_db["entropy"] > 2.4 and word_db["totalCount"] >= 60000) or word_db["totalCount"] <= 50:
        return False
    else:
        return True


def statutes_fenci():
    db = dbutil.get_mongodb_conn()
    statutes_set = db.statutes    # statutes表，没有则自动创建
    for line in statutes_set.find():
        content_words = " ".join(fenci(line["content"]))
        statutes_set.update(
            {"_id": line["_id"]},      # 更新条件
            {'$set': {"contentWords": content_words}},  # 更新内容
            upsert=False,  # 如果不存在update的记录，是否插入
            multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
        )


def case_fenci_first():
    logger = myutil.getLogger("fenci.log")
    db = dbutil.get_mongodb_conn()
    cases_set = db.cases
    words_set = db.words

    for line in cases_set.find({"flag": 1}, no_cursor_timeout=True).batch_size(10):
        logger.info(line["_id"])    # 记录当前xml
        ygsc_words = fenci(line["ygsc"])    # 预处理后的分词结果

        # 1：词长小于指定长度的
        if len(ygsc_words) < 20:
            flag = 0

            # 更新
            cases_set.update(
                {"_id": line["_id"]},  # 更新条件
                {'$set': {"ygscWords": " ".join(ygsc_words),
                          "flag": flag
                          }},  # 更新内容
                upsert=False,  # 如果不存在update的记录，是否插入
                multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
            )
        else:
            # 未处理的分词
            words = psg.cut(line["ygsc"])
            ygsc_words_ori = []
            for (w, flag) in words:
                ygsc_words_ori.append(w)

            r = random.random()
            if r < 0.5:    # train
                flag = 2
                # 2：训练集计算词的信息熵
                ygsc_words_set = set(ygsc_words)
                for word in ygsc_words_set:
                    word_db = words_set.find_one({"_id": word})
                    if word_db is None:  # 新词
                        words_set.insert_one(
                            {
                                "_id": word,
                                "totalCount": 1,
                                "ayCount": {str(line["aydm"]): 1}
                            }
                        )
                    else:
                        if str(line["aydm"]) in word_db["ayCount"]:
                            ay_count = word_db["ayCount"][str(line["aydm"])] + 1
                        else:  # 新案由
                            ay_count = 1
                        ay_name = "ayCount."+str(line["aydm"])
                        words_set.update(
                            {"_id": word},
                            {'$set': {ay_name: ay_count},
                             '$inc': {"totalCount": 1}},
                            upsert=False,  # 如果不存在update的记录，是否插入
                            multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
                        )

            elif r < 0.9:  # test
                flag = 3
            else:          # trial
                flag = 4

            cases_set.update(
                {"_id": line["_id"]},  # 更新条件
                {'$set': {"ygscWordsOrigin": " ".join(ygsc_words_ori),
                          "ygscWords": " ".join(ygsc_words),
                          "flag": flag
                          }},  # 更新内容
                upsert=False,  # 如果不存在update的记录，是否插入
                multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
            )


def case_fenci_second():
    db = dbutil.get_mongodb_conn()
    cases_set = db.cases
    words_set = db.words
    for line in cases_set.find({"flag": {"$ne": 0}, "ygscWords2": {"$exists": False}}, no_cursor_timeout=True).batch_size(10):
        flag = line["flag"]
        ygsc_words = line["ygscWords"].split(" ")
        ygsc_words_2 = []
        # 1：进行词筛选处理
        for word in ygsc_words:
            # 1.1 非停用词和低频词、如果非训练集，还要把未出现的词删掉
            word_db = words_set.find_one({"_id": word})
            if word_db is not None and __not_stopwords(word_db):
                # 1.2: 连续五个词中未重复
                found = False
                end = len(ygsc_words_2)
                start = max(0, end - 5)
                for i in range(start, end):
                    if ygsc_words_2[i] == word:
                        found = True
                        break
                if not found:
                    ygsc_words_2.append(word)

        # 2：处理后词长小于指定长度的
        if len(ygsc_words_2) < 20:
            flag = 0

        cases_set.update(
            {"_id": line["_id"]},  # 更新条件
            {'$set': {"ygscWords2": " ".join(ygsc_words_2),
                      "flag": flag
                      }},  # 更新内容
            upsert=False,  # 如果不存在update的记录，是否插入
            multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
        )


if __name__ == "__main__":
    # statutes_fenci()
    # case_fenci_first()
    # compute_words_entropy()
    # print_stopwords()
    case_fenci_second()