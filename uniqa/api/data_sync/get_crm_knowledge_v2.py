import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import logging
import time
import datetime
import re
import string
import json
import traceback
from zhon.hanzi import punctuation
import pymysql
from sqlalchemy import create_engine

# 导入本地库
import sys
sys.path.append(r"../../")
from configs.config import paths


"""获取【知识平台】QA数据对（通用&机器人）
"""

engine = create_engine(
    f"mysql+pymysql://{username}:{password}@{host}:3306/da_defeat_act"
)
engine = pymysql.connect(
    host, user, password, port=3306,
)

# # 查看有多少个数据库
# sql_query = f"""show databases"""

# # 列出数据库中的所有表
# sql_query = f"""USE saos_crm;
# SHOW TABLES LIKE 'crm_knowledge_base%';"""

# # 查看数据表的行数 18172
# sql_query = f"""
# SELECT COUNT(*) 
# FROM saos_crm.crm_knowledge_base_question
# """
# print(pd.read_sql(sql_query, engine))
# exit()

# sql_query = f"""
# SELECT *
# FROM da_business_sync.crm_knowledge_base_question_answer
# """
# df = pd.read_sql(sql_query, engine)
# print("知识平台答案表 shape: ", df.shape)   # (15559, 13)
# for k,v in df.iloc[0, :].items():
#     print(k,v)
# exit()

# # 查看列名及其注释信息
# sql_query = f"""
# SELECT 
#   TABLE_NAME as '表名',
#   COLUMN_NAME as '列名',
#   COLUMN_COMMENT as '列的注释'
# FROM 
#   information_schema.COLUMNS 
# WHERE 
#   table_schema = 'da_defeat_act' 
#   AND table_name = 'crm_knowledge_base_question_answer_v2'
# """
# print(pd.read_sql(sql_query, engine))
# exit()

# # 查看数据(答案为空)
# sql_query = f"""
# SELECT *
# FROM saos_crm.crm_knowledge_base_question
# WHERE 
#     status=1 
#     and long_effective=1
#     and approval_status=2
#     and question_content like '%金属漆是什么%'
# """
# df = pd.read_sql(sql_query, engine)
# for i,row in df.iterrows():
#     for k,v in row.items():
#         print(k, v)
# exit()


sql = """
SELECT 
    *
FROM 
    database.table
WHERE 
    status = 1
    AND x REGEXP JSON_QUOTE(%s)
"""

params = ("robot|rag_default|common", )  # 挑选文本机器人使用到的知识分类，末尾逗号不可省略
ret = pd.read_sql(sql, engine, params=params)

