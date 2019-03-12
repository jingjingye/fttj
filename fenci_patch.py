# coding=utf-8

import util.db_util as dbutil
import util.function_util as myutil


def case_fenci_patch():
    logger = myutil.getLogger("fenci_patch.log")
    db = dbutil.get_mongodb_conn()
    cases_set = db.cases
    words_set = db.words

    # for line in cases_set.find({"flag": {"$ne": 0}}, no_cursor_timeout=True).batch_size(10):
    #
    #     # 未处理前结果
    #     words = psg.cut(line["ygsc"])
    #     ygsc_words_ori = []
    #     for (w, flag) in words:
    #         ygsc_words_ori.append(w)
    #
    #     cases_set.update(
    #         {"_id": line["_id"]},  # 更新条件
    #         {'$set': {"ygscWordsOrigin": " ".join(ygsc_words_ori)}},  # 更新内容
    #         upsert=False,  # 如果不存在update的记录，是否插入
    #         multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
    #     )

    for line in cases_set.find({"flag": 2, "patch": {"$exists": False}}, no_cursor_timeout=True).batch_size(10):
        logger.info(line["_id"])  # 记录当前xml
        ygsc_words = line["ygscWords"].split(" ")  # 处理后分词
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
                ay_name = "ayCount." + str(line["aydm"])
                words_set.update(
                    {"_id": word},
                    {'$set': {ay_name: ay_count},
                     '$inc': {"totalCount": 1}},
                    upsert=False,  # 如果不存在update的记录，是否插入
                    multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
                )

        cases_set.update(
            {"_id": line["_id"]},  # 更新条件
            {'$set': {"patch": 0}},  # 更新内容
            upsert=False,  # 如果不存在update的记录，是否插入
            multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
        )


def case_fenci_second_patch():
    logger = myutil.getLogger("fenci_patch.log")
    db = dbutil.get_mongodb_conn()
    cases_set = db.cases
    statutes_set = db.statutes

    for line in cases_set.find({"flag": 10, "patch": {"$exists": True}}, no_cursor_timeout=True).batch_size(10):
        logger.info(line["_id"])    # 记录当前xml
        ygsc_words_2 = line["ygscWords2"].split(" ")

        if 3 < len(ygsc_words_2) <= 80:
            cases_set.update(
                {"_id": line["_id"]},  # 更新条件
                {'$set': {"flag": 2}},  # 更新内容
                upsert=False,  # 如果不存在update的记录，是否插入
                multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
            )
            for ftid in line["ftids"]:
                statutes_set.update(
                    {"_id": ftid},
                    {'$inc': {"trainCount": 1}},
                    upsert=False,  # 如果不存在update的记录，是否插入
                    multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
                )


def case_fenci_second_patch_test():
    logger = myutil.getLogger("fenci_patch.log")
    db = dbutil.get_mongodb_conn()
    cases_set = db.cases

    for line in cases_set.find({"flag": 10}, no_cursor_timeout=True).batch_size(10):
        logger.info(line["_id"])  # 记录当前xml
        ygsc_words_2 = line["ygscWords2"].split(" ")

        if 3 < len(ygsc_words_2) <= 80:
            cases_set.update(
                {"_id": line["_id"]},  # 更新条件
                {'$set': {"flag": 4}},  # 更新内容
                upsert=False,  # 如果不存在update的记录，是否插入
                multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
            )


def sampling_train(total_num=10000):
    logger = myutil.getLogger("sample.log")
    db = dbutil.get_mongodb_conn()
    cases_set = db.cases
    statutes_set = db.statutes

    num = 0

    for line in cases_set.find({"flag": 12}, no_cursor_timeout=True).batch_size(10):
        logger.info(line["_id"])    # 记录当前xml
        ygsc_words_2 = line["ygscWords2"].split(" ")

        if 10 < len(ygsc_words_2) < 30:
            num += 1
            cases_set.update(
                {"_id": line["_id"]},  # 更新条件
                {'$set': {"flag": 2}},  # 更新内容
                upsert=False,  # 如果不存在update的记录，是否插入
                multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
            )
            for ftid in line["ftids"]:
                statutes_set.update(
                    {"_id": ftid},
                    {'$inc': {"sampleTrainCount": 1}},
                    upsert=False,  # 如果不存在update的记录，是否插入
                    multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
                )
        if num == total_num:
            break


def sampling_test(total_num=1000):
    logger = myutil.getLogger("sample_test.log")
    db = dbutil.get_mongodb_conn()
    cases_set = db.cases
    statutes_set = db.statutes

    num = 0

    for line in cases_set.find({"flag": 14}, no_cursor_timeout=True).batch_size(10):
        logger.info(line["_id"])    # 记录当前xml
        ygsc_words_2 = line["ygscWords2"].split(" ")

        if 10 < len(ygsc_words_2) < 30:
            ftlegal = True
            for ftid in line["ftids"]:
                statute_db = statutes_set.find_one({"_id": ftid, "sampleTrainCount": {"$exists": True}})
                if statute_db is None:
                    ftlegal = False
                    break
            if ftlegal:
                num += 1
                cases_set.update(
                    {"_id": line["_id"]},  # 更新条件
                    {'$set': {"flag": 4}},  # 更新内容
                    upsert=False,  # 如果不存在update的记录，是否插入
                    multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
                )

        if num == total_num:
            break


if __name__ == "__main__":
    # case_fenci_patch()
    # case_fenci_second_patch()
    # case_fenci_second_patch_test()
    sampling_train(50000)
    # sampling_test(5000)
