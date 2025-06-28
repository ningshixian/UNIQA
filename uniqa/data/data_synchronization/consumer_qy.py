import json
from kafka import KafkaConsumer
import traceback
import time
from collections import defaultdict
import pandas as pd
import requests
import threading
# from multiprocessing import Process
ff = lambda x: x.strip('\n').strip()

"""
# pip install kafka-python python-snappy
# https://github.com/dpkp/kafka-python?tab=readme-ov-file
# https://kafka-python.readthedocs.io/en/master/apidoc/KafkaConsumer.html
"""

def write_dict_to_json(data, file_name):
    # 存储kafka消息
    fw = open(file_name, 'a', encoding='utf-8')
    json_data = json.dumps(data, indent=4, ensure_ascii=False)
    fw.write(json_data)
    fw.write('\n')
    fw.close()


def write_intent_to_onetouch(intent2param, onetouch_path):
    # 1、把同步的意图数据-覆盖更新qa_qy_onetouch.csv
    rows = []
    intent_dict = intent2param
    for intent_id, content in intent_dict.items():
        rows.append([content["vocab"]["knowledgeId"], ff(content['primaryName']),'', '', '','一触即达'])
        for sim_q in content["vocab"]['similary']:    # 暂不考虑相似问
            rows.append([content["vocab"]["knowledgeId"], ff(sim_q),'', '', '','一触即达'])
    df = pd.DataFrame(rows, columns=['question_id', 'question_content', 'answer_content', 'base_name', 'car_type', 'source'])
    df.to_csv(onetouch_path)


def write_entity_to_slot(ent2vocab, slot_path):
    # 2、将枚举实体(entityType=1)写入slot.json
    slot2vocab = {}
    entity_dict = ent2vocab
    with open(slot_path, 'w', encoding='utf-8') as fw:
        for eid,val in entity_dict.items():
            if val["entityType"]==1:    # 枚举
                for entity, ev_list in val['vocab'].items():
                    entity2 = val['entityName'] + '-' + entity
                    slot2vocab[entity2] = ev_list + [entity]
        json_data = json.dumps(slot2vocab, indent=4, ensure_ascii=False)
        fw.write(json_data)


def update_faiss_4_faq():
    # 3、请求本地更新服务
    #   增量更新faiss索引，但无法实现删除向量，遂放弃✖
    #   借助consumer.poll()方法提供的超时时间timeout_ms来控制更新频率，通过update接口更新全局变量
    print("faq向量更新......")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    url = "http://127.0.0.1:8098/update_faq_onetouch"
    headers = {"Content-Type": "application/json"}
    response = requests.request("GET", url, headers=headers)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    print("【faq向量更新】完成!\n")


import os
env = os.getenv("FAQ_ENV", "")

# broker：消息服务器（生产环境）
bootstrap_servers = ['172.21.30.206:9092','172.21.30.207:9092','172.21.30.208:9092']
topic = 'xx_yy_zz'	# 要订阅的topic
bot_id_needed = "18587....."
client_id = "kafka-python-nsx-v2.0.0"
kafka_data_folder = "..."


# group_id = "li-group" # 消费者组
# # 指定group_id：可以让多个消费者协作，每条消息只能被消费1次，实现断点续传；
# # 指定groupid且第一次执行时，auto_offset_reset="earliest"会从最早的数据开始消费。
# # 后续同一个groupid再次执行，则不再读取已消费过的数据，只能消费后续新写入的数据。
    
group_id = None # 消费者组
# 不指定group_id：则进行广播消费，即任一条消息被多次消费
# auto_offset_reset="earliest"，每次都从最早的数据开始消费,重复消费之前的数据;
# auto_offset_reset="latest"，等待后续新写入时再被消费，不会出现数据缺失的情况。

# 创建一个Kafka Consumer实例
consumer = KafkaConsumer(
    topic,	# 要订阅的topic
    bootstrap_servers=bootstrap_servers,
    client_id=client_id,    # 客户端的名称
    group_id=group_id,
    value_deserializer=lambda m: json.loads(m.decode('utf-8')), 
    auto_offset_reset='latest',     # 第一次poll从'earliest'开始消费，之后从'latest'开始消费
    max_poll_records=500,  # 每次poll最大的记录数
    # max_poll_interval_ms=30000,  # 两次间隔poll最大间隔时间ms
    # heartbeat_interval_ms=10000,  # 每次心跳间隔ms
    # session_timeout_ms=20000,  # 心跳会话超时时间ms
    enable_auto_commit=False    # 消费完是否自动回传offset到消息队列
) 

# # 订阅多个 topic
# consumer.subscribe(pattern='topic1, topic2, topic3')

# # StopIteration if no message after 1sec
# KafkaConsumer(consumer_timeout_ms=1000)

# 获取指定topic的分区列表
available_partitions = consumer.partitions_for_topic(topic)
print(f"可用分区 for topic '{topic}': {available_partitions}")  # 0
# ps：【未指定分区】+【消息未指定key】-> 随机地发送到 topic 内的所有可用分区{0} -> 可保证消费消息的顺序性

# 消费者等待超时时间
# 在15s内如果没有消息可用，返回一个空集合
timeout_ms = 15000

# # 逐条消费消息
# for message in consumer:
#     print(f"Received topic: {message.topic}") 
#     print(f"Received key: {message.key}")
#     print(f"Received message: {message.value}")

# 批量消费消息
tmp_msg_sources = []
while True:
    start_time = time.time()
    messages = consumer.poll(timeout_ms=timeout_ms)  # 超时时间timeout_ms，在15s内如果没有消息可用，返回一个空集合

    # 当且仅当接收到 source=BOT_DEPLOY 且 [未]监听到新消息超过1分钟，则更新faiss
    current_time = time.time()
    time_difference = current_time - start_time # 计算时间差（单位为秒）
    if 'BOT_DEPLOY' in tmp_msg_sources and time_difference*1000 >= timeout_ms:  #
        pass

    # 遍历每个批次的消息字典
    for topic_partition, msg_batch in messages.items():  
        # print(f'Topic: {topic_partition.topic}, Partition: {topic_partition.partition}')    # 0
        for message in msg_batch:
            # print(f"Offset: {message.offset}, Partition: {message.partition}")
            print(f"Received key: {message.key}, Received message: {len(message.value)}")   # None, 3
            try:
                pass
            except Exception as e:
                traceback.print_exc()
                continue
        