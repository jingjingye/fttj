# coding=utf-8

import util.db_util as dbutil
import numpy


def case_ref_num(flag):
    db = dbutil.get_mongodb_conn()
    cases_set = db.cases

    max_num = 0
    min_num = 1000
    ref_sum = 0
    case_num = 0
    for line in cases_set.find({"flag": flag}, {"ftids": 1}, no_cursor_timeout=True).batch_size(10):
        ref_num = len(line["ftids"])
        ref_sum += ref_num
        case_num += 1
        if ref_num > max_num:
            max_num = ref_num
        if ref_num < min_num:
            min_num = ref_num

    print("max:%d, min:%d, avg:%f, num:%d" % (max_num, min_num, (ref_sum/case_num), case_num))


def statute_refed_num():
    db = dbutil.get_mongodb_conn()
    cases_set = db.cases

    statutes_vec = numpy.zeros(1405)

    for line in cases_set.find({"flag": 4}, {"label": 1}, no_cursor_timeout=True).batch_size(10):
        label = line["label"]
        statutes_vec += label

    statutes_vec = statutes_vec[numpy.nonzero(statutes_vec)]
    print("max:%f, min:%f, avg:%f, num:%d" % (numpy.max(statutes_vec), numpy.min(statutes_vec), numpy.average(statutes_vec), statutes_vec.shape[0]))


if __name__ == "__main__":
    # case_ref_num(2)
    statute_refed_num()