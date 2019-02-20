# coding=utf-8

import jieba.posseg as psg
import util.db_util as dbutil
import util.function_util as myutil


def case_fenci_patch():
    logger = myutil.getLogger("fenci_patch.log")
    db = dbutil.get_mongodb_conn()
    cases_set = db.cases
    words_set = db.words

    for line in cases_set.find({"flag": {"$ne": 0}}, no_cursor_timeout=True).batch_size(10):
        logger.info(line["_id"])  # 记录当前xml

        # 未处理前结果
        words = psg.cut(line["ygsc"])
        ygsc_words_ori = []
        for (w, flag) in words:
            ygsc_words_ori.append(w)

        if line["flag"] == 2:       # DF补丁
            ygsc_words = line["ygscWords"].split(" ")   # 处理后分词
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
            {'$set': {"ygscWordsOrigin": " ".join(ygsc_words_ori)}},  # 更新内容
            upsert=False,  # 如果不存在update的记录，是否插入
            multi=False,  # 可选，mongodb 默认是false,只更新找到的第一条记录
        )


if __name__ == "__main__":
    case_fenci_patch()