# coding=utf-8

import util.db_util as dbutil
import util.function_util as myutil
import model.fptree as fptree


def trainRules():
    '''
    挖掘关联规则
    :return:
    '''
    config = myutil.read_config("conf/fttj.conf")
    # 获取连接
    db = dbutil.get_mongodb_conn()
    cases_set = db.cases
    rules_set = db.rules

    # 挖掘规则
    ftids = [case["ftids"] for case in cases_set.find({"flag": 2}, {"ftids":1}, no_cursor_timeout=True)]
    rules = fptree.generateRules(ftids, config["minsup"], config["minconf"])
    # 简化规则
    rules = __prunedRules(rules)

    # 存储规则
    for rule in rules:
        rules_set.insert({
            "from": list(rule[0]),
            "to": list(rule[1])
        })


def __prunedRules(rules):
    # 保留toItem中最大的
    fromDict = {}
    for rule in rules:
        if rule[0] in fromDict:
            toList = fromDict[rule[0]]
            i = 0
            insert = True
            while i < len(toList):
                if toList[i].issubset(rule[1]):
                    del toList[i]
                    i -= 1
                elif rule[1].issubset(toList[i]):
                    insert = False
                    break
                i += 1
            if insert:
                toList.append(rule[1])
        else:
            fromDict[rule[0]] = [rule[1]]

    # 保留fromItem中最小的
    toDict = {}
    for (fromItem, toList) in fromDict.items():
        for toItem in toList:
            if toItem in toDict:
                fromList = toDict[toItem]
                i = 0
                insert = True
                while i < len(fromList):
                    if fromItem.issubset(fromList[i]):
                        del fromList[i]
                        i -= 1
                    elif fromList[i].issubset(fromItem):
                        insert = False
                        break
                    i += 1
                if insert:
                    fromList.append(fromItem)
            else:
                toDict[toItem] = [fromItem]

    # 转化输出
    rltList = []
    for (toItem, fromList) in toDict.items():
        for fromItem in fromList:
            rltList.append((fromItem, toItem))

    return rltList


def runRules(oriStatutes):
    # 得到包含的rules
    db = dbutil.get_mongodb_conn()
    rules_set = db.rules
    rules = rules_set.find({"from": {"$in": oriStatutes}})

    # 遍历得到伴随
    oriSet = set(oriStatutes)
    assoRules = set(oriStatutes)
    for rule in rules:
        if set(rule["from"]).issubset(oriSet):
            assoRules.update(rule["to"])
    return list(assoRules)