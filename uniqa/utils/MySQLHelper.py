#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# https://blog.csdn.net/kk185800961/article/details/78635460
import pymysql
from DBUtils.PooledDB import PooledDB  # 导入线程池对象

"""
mysql操作类，对mysql数据库进行增删改查
"""


def read_config(config_file):
    """
    读取数据库配置文件 config.ini
    """
    from configparser import RawConfigParser
    from configparser import ConfigParser
    
    CFG = RawConfigParser()
    CFG.read(config_file, encoding="utf-8")
    # section_list = CFG.sections()
    return CFG


class PooledDBConnection(object):
    """MySQL数据库的缓存连接池
    参考：《python DbUtils 使用教程》https://cloud.tencent.com/developer/article/1568031
    参考：《python数据库连接工具DBUtils》https://segmentfault.com/a/1190000017952033
    """
    
    def __init__(self, DB_CONFIG):
        """DB_CONFIG : 数据库配置信息"""
        self.pool = PooledDB(
            creator=pymysql,
            host=DB_CONFIG.get("host"),
            port=int(DB_CONFIG.get("port")),
            user=DB_CONFIG.get("user"),
            password=DB_CONFIG.get("passwd"),
            db=DB_CONFIG.get("db"),
            charset="utf8", # 数据库连接编码
            mincached=1,  # 连接池中空闲连接的初始数量(0表示不创建初始空闲连接) 1
            maxcached=12,  # 连接池中允许的最大空闲连接数(0或None表示无限制)
            maxconnections=0,  # 允许的最大连接数(0或None表示无限制)    5
            blocking=True,  # 连接池中如果没有可用连接后，是否阻塞等待。True，等待；False，不等待然后报错
            ping=4, # 4 = when a query is executed
        )   # MySQL连接池

    def get_conn(self):
        """
        连接数据库
        :return: conn, cursor
        """
        try:
            conn = self.pool.connection()
            cursor = conn.cursor()  # pymysql.cursors.DictCursor
        except Exception as e:
            print("数据库连接失败========, " + str(e))
            exit()
        return conn, cursor

    def reset_conn(self, conn, cursor):
        """
        :param conn: 数据库连接
        :param cursor: 数据库指针
        :return: Null
        """
        try:
            cursor.close()  # 关闭游标
            conn.close()    # 关闭连接（连接会返回到连接池让后续线程继续使用）
        except Exception as err:
            raise ("MySQL关闭异常: ", str(err))

    # 执行查询
    def ExecQuery(self, sql, values=None):
        res = ""
        try:
            conn, cursor = self.get_conn()
            cursor.execute(sql, values)  # 防止SQL注入攻击
            res = cursor.fetchall()
            self.reset_conn(conn, cursor)
        except Exception as e:
            raise Exception("连接或查询失败：" + str(e))
        return res

    # 执行非查询类语句
    def ExecNonQuery(self, sql, values=None):
        flag = False
        # self.pool.ping(reconnect=True)
        try:
            conn, cursor = self.get_conn()
            conn.begin()   # 开始事务
            cursor.execute(sql, values) # 防止SQL注入攻击
            conn.commit()   # 提交事务
            self.reset_conn(conn, cursor)
            flag = True
        except Exception as e:
            conn.rollback()   # 回滚事务
            raise Exception("连接或执行失败: " + str(e))
        return flag


class PyMysqlConnection:
    def __init__(self):
        self.db = None
        self.cursor = None

    def __del__(self):
        # 关闭数据库连接
        self.db.close()

    def create_db(self, server):
        try:
            self.db = pymysql.connect(
                host=server.get("host"),
                port=server.getint("port"),
                user=server.get("user"),
                passwd=server.get("passwd"),
                db=server.get("db"),
                charset="utf8",
            )
            self.cursor = self.db.cursor()
            print("数据库连接成功~")
        except Exception as e:
            print("数据库连接失败！" + str(e))
            exit()

    def query(self, sql):
        """SELECT * FROM infor;"""
        project = {}
        self.db.ping(reconnect=True)
        self.cursor.execute(sql)  # 执行 SQL查询
        result = self.cursor.fetchall()
        for row in result:
            project[row[0]] = row[1]
        return project
    
    # 执行非查询类语句
    def ExecNonQuery(self, sql, autoclose=False):
        # 检查MySQL连接是否还在，不存在的话就重连
        self.db.ping(reconnect=True)
        flag = False
        try:
            with self.db.cursor() as cursor:  # 查询游标
                cursor.execute(sql)
            self.db.commit()
            if autoclose:
                self.close()
            flag = True
        except Exception as err:
            self.db.rollback()
            print("执行失败, %s" % err)
        return flag

    def close(self):
        if self.db:
            try:
                self.db.close()
                print("MySQL连接已关闭")
            except Exception as err:
                raise ("MySQL关闭异常: ", str(err))
