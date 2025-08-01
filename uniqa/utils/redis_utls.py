import os
import json
import redis
from redis.sentinel import Sentinel

# from common import constants, apollo_config


# 高可用版 redis 哨兵连接
class RedisUtilsSentinel(object):
    def __init__(self, redis_sentinel_config):
        self.redis_sentinel_config = redis_sentinel_config
        # 获取 redis sentinel ip 信息
        self.sentinel_list = []
        for sentinel_ip in self.redis_sentinel_config["redis_sentinel_list"]:
            ip, port = sentinel_ip.split(":")
            self.sentinel_list.append((ip, port))

        # 加载 redis sentinel
        self.sentinel = Sentinel(
            self.sentinel_list,
            min_other_sentinels=0,
            sentinel_kwargs={
                "password": self.redis_sentinel_config["redis_password"]
            },
        )

        # 初始化 master
        self.master = self.sentinel.master_for(
            self.redis_sentinel_config["sentinel_master"],
            db=0,
            password=self.redis_sentinel_config["redis_password"],
            decode_responses=True,
        )

        # 初始化 slave
        self.slave = self.sentinel.slave_for(
            self.redis_sentinel_config["sentinel_master"],
            db=0,
            password=self.redis_sentinel_config["redis_password"],
            decode_responses=True,
        )

        # 使用master进行写的操作,使用slave进行读的操作

        # # 检查连接
        self.master.ping()
        self.slave.ping()

    def get_redis_client(self):
        return self

    def get(self, key):
        val = self.slave.get(key)
        return json.loads(val) if val else None

    def set(self, key, val):
        return self.master.set(
            key, json.dumps(val, ensure_ascii=False), ex=2*24*60*60
        )
        # ex：过期时间（秒），时间到了后redis会自动删除，暂定 2 天

    def delete(self, key):
        return self.master.delete(key)
