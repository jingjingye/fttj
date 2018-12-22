# coding=utf-8

from pymongo import MongoClient
import util.function_util as myutil

db = None


def get_mongodb_conn():
    global db
    if db is None:
        config = myutil.read_config("conf/fttj.conf")
        conn = MongoClient(config["mongodb_uri"])
        db = conn.lawyjj  # 连接lawyjj数据库，没有则自动创建
    return db
