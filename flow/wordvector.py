# coding=utf-8

from gensim.models import word2vec
import numpy as np
import os
import re
import xml.dom.minidom
import pickle
import util.function_util as myutil
import util.db_util as dbutil
from data_process import fenci

# 低频词 20
vocabulary = None
word_embedding = None


def trainWordvector(genFile=False, corpusFile="checkpoint/corpus.txt"):
    config = myutil.read_config("conf/fttj.conf")
    logger = myutil.getLogger("parsexml.log")

    # 1: 生成corpus文档
    if genFile:
        with open(corpusFile, 'w', encoding="utf-8") as corpus:
            # 案件
            dir = config["corpus_dir"]
            for file in os.listdir(dir):
                try:
                    dom = xml.dom.minidom.parse(dir + '/' + file)
                    nodelist = dom.documentElement.getElementsByTagName("AJJBQK")
                    if len(nodelist) > 0:
                        text = nodelist[0].getAttribute("value")
                        __appendToFile(text, corpus)
                except xml.parsers.expat.ExpatError:
                    logger.error("%s编码错误" % file)

            # 法条
            db = dbutil.get_mongodb_conn()
            statutes_set = db.statutes
            for line in statutes_set.find():
                __appendToFile(line["content"], corpus)

    # 2: 训练词向量
    sentences = word2vec.LineSentence(corpusFile)
    model = word2vec.Word2Vec(sentences, min_count=20, size=config["embedding_size"])
    vocabulary, word_embedding = __get_word_vector(model.wv)

    # 3: 保存模型
    with open("checkpoint/vocabulary.pk", "wb") as file:
        pickle.dump(vocabulary, file)
    with open("checkpoint/wordvector.pk", "wb") as file:
        pickle.dump(word_embedding, file)


def __appendToFile(text, file):
    sentences = re.split("。|！|？", text)
    for sentence in sentences:
        words = fenci(sentence)
        if len(words) > 0:
            file.write(" ".join(words)+"\n")
            file.flush()


def __get_word_vector(wv):
    vocabulary = dict(zip(wv.index2word, range(2, len(wv.index2word) + 2)))
    vocabulary['<padding>'] = 0
    vocabulary['<unk>'] = 1
    word_embedding = wv.syn0
    # 把补足的部分记为0
    word_zero = np.zeros(shape=[word_embedding.shape[1]], dtype=np.float32)
    # 把未知的<unk>初始化为所有词向量的均值
    word_mean = np.mean(word_embedding, axis=0)
    word_embedding = np.vstack([word_zero, word_mean, word_embedding])
    return vocabulary, word_embedding


def load_vocabulary():
    global vocabulary
    if vocabulary is None:
        with open("checkpoint/vocabulary.pk", "rb") as file:
            vocabulary = pickle.load(file)
    return vocabulary


def load_word_embedding():
    global word_embedding
    if word_embedding is None:
        with open("checkpoint/wordvector.pk", "rb") as file:
            word_embedding = pickle.load(file)
    return word_embedding


def seq2id(seq):
    load_vocabulary()
    words = seq.split(" ")
    seq_id = []
    for word in words:
        if word in vocabulary:
            seq_id.append(vocabulary[word])
        else:
            seq_id.append(vocabulary['<unk>'])
    return seq_id