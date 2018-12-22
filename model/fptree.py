# coding:utf-8
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur


def generateRules(dataSet, minSup=1, minConf=0.7):
    '''
    由数据集得到关联规则
    :param dataSet:
    :param minSup:
    :param minConf:
    :return:
    '''
    freqItems = mineFPtree(dataSet, minSup)
    return mineRules(freqItems, minConf)


def mineRules(freqItems, minConf=0.7):
    '''
    由频繁项集得到关联规则
    :param freqItems:
    :param minConf:
    :return:
    '''
    ruleList = []
    for items in freqItems.keys():
        toItems = [frozenset([item]) for item in items]
        if len(items) >= 2:  # 2个以上的组合
            __rulesFromConseq(items, toItems, freqItems, ruleList, minConf)
        # 1个没有规则
    return ruleList


def __rulesFromConseq(allItems, toItems, freqItems, ruleList, minConf):
    k = len(toItems[0])  # 当前推得的频繁项集的长度
    if len(allItems) > k:  # 若差值大于0， 还可以继续推
        prunedToItems = __calcConf(allItems, toItems, freqItems, ruleList, minConf)
        if len(prunedToItems) > 1 and len(allItems) > k+1:
            nextToItems = __aprioriGen(prunedToItems, k + 1)
            if len(nextToItems) > 0:
                __rulesFromConseq(allItems, nextToItems, freqItems, ruleList, minConf)
            

def __calcConf(items, toItems, freqItems, ruleList, minConf):
    prunedToItems = []
    for toItem in toItems:
        fromItem = frozenset(items - toItem)
        conf = freqItems[items] / freqItems[fromItem]
        if conf >= minConf:
            # print("{0} --> {1} conf:{2}".format(fromItem, toItem, conf))
            ruleList.append((fromItem, toItem, conf))
            prunedToItems.append(toItem)
    return prunedToItems


def __aprioriGen(Lk_1, k):
    '''
    由Lk-1生成Lk (k代表频繁项的元素个数)
    :param Lk_1:
    :param k:
    :return:
    '''
    retList = []
    lenLk_1 = len(Lk_1)
    for i in range(lenLk_1):
        for j in range(i + 1, lenLk_1):
            L1 = list(Lk_1[i])[:k - 2]    # 第i个item的前k-2位
            L2 = list(Lk_1[j])[:k - 2]    # 第j个item的前k-2位
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk_1[i] | Lk_1[j])   # 如果前k-2位相同，取并集，合并后有k位
    return retList


def mineFPtree(dataSet, minSup=1):
    '''
    FP树获取频繁项集
    :param dataSet:
    :param minSup:
    :return:
    '''
    initSet = __createInitDict(dataSet)
    fptree, headerTable = __createFPtree(initSet, minSup)
    freqItems = {}
    __subMineFPtree(headerTable, minSup, set(), freqItems)
    return freqItems


def __subMineFPtree(preHead, minSup, prePath, freqItems):
    # 最开始的频繁项集是headerTable中的各元素
    items = [(v[0], v[1][0]) for v in sorted(preHead.items(), key=lambda p: p[1][0])]  # 根据频繁项的总频次排序
    for curItem, count in items:  # 对headerTable中的每个元素
        curPath = prePath.copy()
        curPath.add(curItem)
        freqItems[frozenset(curPath)] = count
        condPattBases = __findPrefixPath(curItem, preHead)  # 当前频繁项集的条件模式基
        _, curHead = __createFPtree(condPattBases, minSup)  # 构造当前频繁项的条件FP树
        if curHead != None:
            __subMineFPtree(curHead, minSup, curPath, freqItems)  # 递归挖掘条件FP树


def __findPrefixPath(item, headerTable):
    '''
    获取条件模式基
    :param item:
    :param headerTable:
    :return:
    '''
    treeNode = headerTable[item][1]  # basePat在FP树中的第一个结点
    condPats = {}
    while treeNode != None:
        prefixPath = []
        __ascendFPtree(treeNode, prefixPath)  # prefixPath是倒过来的，从treeNode开始到根
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count  # 关联treeNode的计数
        treeNode = treeNode.nodeLink  # 下一个basePat结点
    return condPats


# 回溯
def __ascendFPtree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        __ascendFPtree(leafNode.parent, prefixPath)


def __createFPtree(dataSet, minSup):
    '''
    生成FP树
    :param dataSet:
    :param minSup:
    :return:
    '''
    headerTable = {}
    # 1.为每个元素计数
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]

    # 2.删除不满足最小支持度的元素
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del (headerTable[k])
        else:     # 3.建立元素表(包含元素count和元素出现链表)
            headerTable[k] = [headerTable[k], None]  # element: [count, node]
    freqItemSet = set(headerTable.keys())  # 满足最小支持度的元素集合
    if len(freqItemSet) == 0:
        return None, None

    retTree = treeNode('Root Node', 1, None)
    for tranSet, count in dataSet.items():
        # dataSet：[element, count]
        localD = {}
        for item in tranSet:
            if item in freqItemSet:  # 过滤，只取该样本中满足最小支持度的频繁项
                localD[item] = headerTable[item][0]  # element : count
        if len(localD) > 0:
            # 根据全局频数降序、item数字升序(若为字符串，则为字典序升序)对单样本排序
            # orderedItem = [v[0] for v in sorted(localD.items(), key=lambda p:(p[1], -ord(p[0])), reverse=True)]
            orderedItem = [v[0] for v in sorted(localD.items(), key=lambda p: (p[1], -int(p[0])), reverse=True)]
            # 用过滤且排序后的样本更新树
            __updateFPtree(orderedItem, retTree, headerTable, count)
    return retTree, headerTable


def __updateFPtree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        # 判断items的第一个结点是否已作为子结点
        inTree.children[items[0]].inc(count)
    else:
        # 创建新的分支
        curNode = treeNode(items[0], count, inTree)
        inTree.children[items[0]] = curNode
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = curNode
        else:
            __updateHeader(headerTable[items[0]][1], curNode)
    # 递归
    if len(items) > 1:
        __updateFPtree(items[1:], inTree.children[items[0]], headerTable, count)


def __updateHeader(nodeToTest, targetNode):
    '''
    更新元素node链
    :param nodeToTest:
    :param targetNode:
    :return:
    '''
    while nodeToTest.nodeLink != None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def __createInitDict(dataSet):
    '''
    将相同的项集合并计数
    :param dataSet:
    :return:
    '''
    retDict = {}
    for trans in dataSet:
        key = frozenset(trans)
        if key in retDict:
            retDict[frozenset(trans)] += 1
        else:
            retDict[frozenset(trans)] = 1
    return retDict