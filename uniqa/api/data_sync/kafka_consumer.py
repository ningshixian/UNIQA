import os, sys
import signal
import json
import traceback
import time
from collections import defaultdict
import pandas as pd
import requests
import threading
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum
import codecs
import datetime
from uniqa.logging import logDog as logger

# 导入 kafka-python 的相关模块
from kafka import KafkaConsumer as KPConsumer  # 使用别名以避免与我们的类名冲突
from kafka.structs import TopicPartition
from kafka.errors import KafkaError

sys.path.append(r"../../")
from utils import RedisUtilsSentinel
from configs.config import paths
from configs.config import kafka_config, redis_config, snapshot_key

"""
# pip install kafka-python python-snappy
# https://github.com/dpkp/kafka-python?tab=readme-ov-file
# https://kafka-python.readthedocs.io/en/master/apidoc/KafkaConsumer.html
"""


# --- 使用枚举代替魔法字符串 ---
class SourceType(str, Enum):
    ENTITY = 'ENTITY'
    PARAM = 'PARAM'
    INTENT = 'INTENT'
    BOT_DEPLOY = 'BOT_DEPLOY'

class OperateType(str, Enum):
    INSERT = 'INSERT'
    UPDATE = 'UPDATE'
    DELETE = 'DELETE'
    # DEPLOY = 'DEPLOY'

# 常量定义
SNAPSHOT_INTERVAL_SECONDS = 3600  # 快照间隔，3600秒 = 1小时
DEFAULT_CONSUMER_WAIT_TIME = 60*1000        # 消费者等待超时时间(ms) 60s=1min
DEFAULT_MAX_POLL_RECORDS = 500            # 每次消费的poll最大记录数


class KnowledgeSyncConsumer:
    """
    一个从 Kafka 消费知识数据，维护内存缓存，并定期创建快照的消费者。
    """

    def __init__(self, kafka_config, redis_conn):
        self.servers = kafka_config.bootstrap_servers  # .split(",")
        self.topic = kafka_config.topic_name
        self.group_id = kafka_config.group_id
        self.client_id = kafka_config.client_id
        self.offset = kafka_config.auto_offset_reset
        self.bot_id_needed = kafka_config.bot_id_needed
        self.redis_conn = redis_conn

        self.last_snapshot_time = 0.0                               # 最后一次的快照时间
        self.consumer: Optional[KPConsumer] = None

        # --- 类封装：qy一触即达数据 将所有状态保存在实例中 ---
        self.knowledge: Dict[str, Any] = {
            "intent2param": {},
            "param2ent": {},
            "ent2vocab": {},
            # "qa_qy_onetouch": [], 
            # "slot": {},
        }
        # 使用枚举作为 key，更安全
        self._source_handlers = {
            SourceType.ENTITY: self._handle_entity,
            SourceType.PARAM: self._handle_param,
            SourceType.INTENT: self._handle_intent,
        }
    
    def _load_snapshot(self) -> bool:
        """
        [内部方法] 启动时尝试从 Redis 快照恢复。
        成功恢复返回 True，否则返回 False。
        """
        if not self.redis_conn:
            logger.warning("Redis connection not provided, cannot load snapshot.")
            return False

        try:
            logger.info("Loaded snapshot. 初次启动，尝试从 Redis 快照恢复...")

            raw = self.redis_conn.get(snapshot_key)
            if not raw:
                logger.info("   No snapshot found in Redis.")
                return False

            snap = json.loads(raw)
            self.knowledge.update(snap.get("data", {}))
            offsets = snap.get("offsets", {})   # 偏移量

            if not offsets:
                logger.warning("    Snapshot found but contains no offset information.")
                return False

            # 手动分配分区和 topic
            tp = TopicPartition(topic=self.topic, partition=0)
            self.consumer.assign([tp])  # 这里是声明我要手动管理这个consumer的这个partition
            logger.info(f"  Assigned partitions manually: {tp}")

            # 为分区设置偏移量 → 最新
            offset_to_seek = offsets.get(str(tp.partition))
            if offset_to_seek is not None:
                self.consumer.seek(tp, offset=offset_to_seek) # 从0区的第offset位置开始读取 
                # self.consumer.seek_to_end(tp)   # latest
                logger.info(f"  Seeking partition {tp.partition} to offset {offset_to_seek}")

            # 将一触即达中的【意图问题和实体】写入本地json文件
            #（重建用于faq召回和槽位提取）
            _qa_qy_onetouch = self._build_onetouch_data(self.knowledge["intent2param"])
            _slot =  self._build_slot_data(self.knowledge["ent2vocab"])
            self._write_json_file(_qa_qy_onetouch, paths.kafka.onetouch)
            self._write_json_file(_slot, paths.kafka.slot)
            logger.info(f"  【意图问题和实体】已写入本地json → {paths.kafka.onetouch} | {paths.kafka.slot}")
            # 备份 knowledge 数据
            self._write_json_file(self.knowledge["intent2param"], paths.kafka.intent)
            self._write_json_file(self.knowledge["param2ent"], paths.kafka.param)
            self._write_json_file(self.knowledge["ent2vocab"], paths.kafka.entity)

            logger.info(f"Loaded snapshot finished.\n")
            return True
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode snapshot JSON from Redis: {e}")
            return False
        except Exception as e: # Redis aof other errors
            logger.error(f"Failed to load snapshot from Redis: {e}", exc_info=True)
            return False

    def _save_snapshot(self):
        """[内部方法] 保存知识库和消费者偏移量到 Redis 快照。"""
        if not self.redis_conn:
            logger.warning("Redis connection not provided, cannot save snapshot.")
            return

        if not self.consumer:
            logger.warning("Consumer not running, cannot save snapshot.")
            return

        try:
            # 获取当前分配到的分区的偏移量
            # consumer.position(tp) 返回下一个要消费的消息的偏移量，这正是我们需要的
            offsets = {
                str(tp.partition): self.consumer.position(tp)
                for tp in self.consumer.assignment()
            }
            if not offsets:
                logger.warning("No partitions assigned, skipping snapshot with offsets.")
                return

            snap = {"data": self.knowledge, "offsets": offsets, "timestamp": time.time()}
            self.redis_conn.set(snapshot_key, json.dumps(snap))
            logger.info(f"Snapshot saved with offsets for {len(offsets)} partitions. offsets: {offsets}")
            self.last_snapshot_time = time.time() # 更新快照时间
        except Exception as e:
            logger.error(f"Failed to save snapshot to Redis: {e}", exc_info=True)
    
    def _handle_entity(self, record: Dict, operateType: OperateType):
        """处理 ENTITY 类型的记录"""
        entId = record.get('entId')
        if not entId: return

        if operateType in [OperateType.INSERT, OperateType.UPDATE]:
            # 实体和实体值是一对多关系
            self.knowledge["ent2vocab"][entId] = {
                'entityName': record.get('entityName', ''),    # 实体名称
                'entityType': record.get('entityType', ''),    # 实体类型（  1-枚举/2-正则/3-意图/4-系统/5-算法实体 ）
                'vocab': {                                     # 实体值字典 {实体值:[同义词]}
                    entry['vocabularyEntryName']: [syn for syn in entry.get('synonyms', '').split('|') if syn]
                    for entry in record.get('entVocabularyEntries', [])   # 实体值列表
                }
            }
        elif operateType == OperateType.DELETE:
            # 使用 .pop(key, None) 更安全，如果 key 不存在不会报错
            self.knowledge["ent2vocab"].pop(entId, None)

    def _handle_param(self, record: List[Dict], operateType: OperateType):
        """处理 PARAM 类型的记录"""
        records = record if isinstance(record, list) else [record]

        for rec in records:
            paramId = rec.get('id')
            if not paramId: continue

            if operateType in [OperateType.INSERT, OperateType.UPDATE]:
                # 原始逻辑中，INSERT/UPDATE 的 record 是一个列表
                self.knowledge["param2ent"][paramId] = {        # 变量和实体是一对一关系
                    'paramName': rec.get('paramName', ''),      # 变量 id
                    'entityId': rec.get('entityId', '')         # 关联实体id
                }
            elif operateType == OperateType.DELETE:
                # 原始逻辑中，DELETE 的 record 是一个字典
                self.knowledge["param2ent"].pop(paramId, None)

    def _handle_intent(self, record: Dict, operateType: OperateType):
        """处理 INTENT 类型的记录"""
        intentId = record.get('id')
        if not intentId: return
        # if not record['status']=="1": return

        if operateType in [OperateType.INSERT, OperateType.UPDATE]:
            diaInteractInput = record.get('diaInteractInput', [])   # 标准问题和相似问题列表 ↓↓↓
            # 使用生成器表达式和 next() 高效查找主问题
            primary_info = next((k for k in diaInteractInput if k.get('knowledgeId') != 0 or k.get("sentenceType") == "1"), None)
            # 使用列表推导式查找相似问
            similar_sentences = [k['sentence'] for k in diaInteractInput if not (k.get('knowledgeId') != 0 or k.get("sentenceType") == "1")]

            # 意图和变量是一对多关系
            # intentId = record.get('intentId', '')     # 意图id
            self.knowledge["intent2param"][intentId] = {
                'intentName': record.get('intentName', ''),     # 意图名称
                'primaryName': record.get('name', ''),          # 意图对应的问法
                'flowId': record.get('flowId', ''),             # 意图关联的流程id
                'flowName': record.get('flowName', ''),         # 意图关联的流程名称
                'paramIdList': record.get('paramIdList', []),   # 意图关联的变量id列表
                'vocab': {  # 意图关联的标准问和相似问
                    'knowledgeId': primary_info['knowledgeId'] if primary_info else None,
                    'primary': primary_info['sentence'] if primary_info else '',
                    # 'markedSentence': primary_info['markedSentence'] if primary_info else '',     # ?
                    # 'markedEntities': primary_info['markedEntities'] if primary_info else '',     # ?
                    'similary': similar_sentences
                }
            }
        elif operateType == OperateType.DELETE:
            self.knowledge["intent2param"].pop(record.get('id'), None)

    def _process_message(self, msg):
        """处理单条 Kafka 消息"""
        try:
            # logger.info(("%s:%d:%d: key=%s value=%s" % (msg.topic, msg.partition, msg.offset, msg.key, msg.value)))
            value_dict = json.loads(msg.value)

            # # 检查字段是否合法(可以更具体)
            # source_str = value_dict.get('source')
            # op_type_str = value_dict.get('operateType')
            # if not all([source_str, op_type_str, 'data' in value_dict]):
            #     logger.warning(f"Message missing required fields: {msg.value}")
            #     return

            record = json.loads(value_dict['data'])  # json消息
            bizTime = value_dict.get('bizTime', '') # 时间戳 timestamp
            source = value_dict.get('source', '').upper() # (意图、变量、实体、BOT_DEPLOY) ✔
            operateType = value_dict.get('operateType', '').upper()  # INSERT UPDATE DELETE BOT_DEPLOY✔
            # logger.info(f"{source} {operateType} | record:{record}")

            # 根据bot_id过滤不相关的消息
            if self.bot_id_needed:
                # 统一 bot_id 获取逻辑
                if source == 'PARAM':
                    # PARAM的新增是批量的，更新和删除是单个的，需统一一下格式
                    if isinstance(record, dict): 
                        record = [record]
                    bot_id = record[0].get('botid', record[0].get('botId', ''))
                else:
                    bot_id = record.get('botId', record.get('botid', ''))
                
                # 根据bot_id过滤不相关的消息（仅针对生产环境）
                if bot_id and not bot_id.startswith(self.bot_id_needed):
                    return # 静默忽略不相关的消息
            
            # # 存储kafka消息
            # write_dict_to_json(value_dict, received_kafka_data_path)

            # # 清洗数据
            # clean_data = self.clean_data(value_dict)
            
            if source == SourceType.BOT_DEPLOY:
                logger.info("source=BOT_DEPLOY -> self.knowledge{意图、变量、实体} save to redis!!!")
                self._save_snapshot()
                self._update_qy_knowledge_4_faq()     # push to API
                # input("Press Enter to continue...")
            else:
                # source in [entity, param, intent]
                logger.info(f"逐条消息处理：根据 source={source} 和 operateType={operateType} 分发任务!")
                handler = self._source_handlers.get(source)
                if handler:
                    handler(record, operateType)
                    # logger.info(f"----{msg.offset}----")
                else:
                    logger.warning(f"Unknown source type: {source}")
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from message value: {msg.value}. Error: {e}")
        except (KeyError, TypeError) as e:
            traceback.print_exc()
            logger.error(f"Data structure error in message: {msg.value}. Error: {e}")
        except Exception:
            traceback.print_exc()
            logger.error(f"Unexpected error processing message at offset {msg.offset}", exc_info=True)

    def _processing_loop(self):
        """主消费循环"""
        logger.info("Consumer started. Waiting for messages loop...")
        while True:
            # --- 优化: 更高效和准确的定时快照逻辑 ---
            # 每隔 1h=3600s 做一次快照（可换成按条数或异步线程）
            if time.time() - self.last_snapshot_time > SNAPSHOT_INTERVAL_SECONDS:
                self._save_snapshot()

            # 使用 poll 获取批量消息，提供超时时间参数timeout_ms来控制更新频率
            # 如果在60s=1min内如果没有消息可用，返回一个空集合
            msg_pack = self.consumer.poll(timeout_ms=DEFAULT_CONSUMER_WAIT_TIME)
            if not msg_pack:
                continue

            # 遍历每个批次的消息字典
            for tp, messages in msg_pack.items():
                if len(messages) > 1:
                    logger.info(f"Processing-loop {len(messages)} messages from partition {tp.partition}\n")
                for msg in messages:
                    self._process_message(msg)

    # --- 优化: 初始化 Kafka consumer，并处理快照加载 ---
    def _init_snap_consumer(self):
        """初始化 Kafka 消费者，并处理快照加载"""
        consumer_config = {
            "bootstrap_servers": self.servers,
            "group_id": self.group_id,
            "client_id": self.client_id,
            "auto_offset_reset": self.offset,  # 'earliest' 或 'latest'
            "enable_auto_commit": False,  # 手动管理偏移量，所以关闭自动提交
            "value_deserializer": lambda v: v.decode("utf-8", "ignore"),
            "key_deserializer": lambda k: k.decode("utf-8", "ignore"),
            "max_poll_records": DEFAULT_MAX_POLL_RECORDS,
            # 根据需要添加 SASL 配置
        }
        self.consumer = KPConsumer(**consumer_config)
        logger.info(f"Kafka consumer initialized successfully for topic: {self.topic}")
        
        # 尝试从快照加载。如果失败，则回退到标准的订阅模式
        if not self._load_snapshot():
            self.consumer.subscribe([self.topic])  # 订阅指定的主题
            logger.info("No snapshot loaded, subscribing to topic to get partitions automatically.")
        
        # 记录快照时间
        self.last_snapshot_time = time.time()

        # # 获取主题的分区信息
        # logger.info(self.consumer.partitions_for_topic(self.topic))  # {0}
        # # # 获取主题列表
        # # logger.info(self.consumer.topics())
        # # 获取当前消费者订阅的主题
        # logger.info(self.consumer.subscription())   # {'sync_lixiang_intent_data'}
        # # 获取当前消费者topic、分区信息
        # logger.info(self.consumer.assignment())     # {TopicPartition(topic='sync_lixiang_intent_data', partition=0)}
        # # 获取当前主题{0}的最新偏移量
        # tp = TopicPartition(topic=self.topic, partition=0)
        # offset = self.consumer.position(tp)  # <class 'int'>
        # logger.info(f"分区0 for topic '{self.topic}' 的最新偏移量: {offset}")
        # # ps：【未指定分区】+【消息未指定key】-> 随机地发送到 topic 内的所有可用分区{0} -> 可保证消费消息的顺序性

    def run(self):
        """启动消费者，包含初始化和主消费循环"""
        try:
            self._init_snap_consumer()

            # 捕获 SIGTERM 信号并优雅退出
            def graceful_shutdown(sig, frame):
                logger.info("Received SIGTERM, saving snapshot and closing consumer...")
                self._save_snapshot()  # 保存快照
                self.consumer.close()  # 关闭消费者
                sys.exit(0)

            signal.signal(signal.SIGTERM, graceful_shutdown)
            signal.signal(signal.SIGINT, graceful_shutdown) # 也处理 Ctrl+C

            self._processing_loop()
        
        except KafkaError as e:
            logger.error(f"A Kafka error occurred: {e}")
        except Exception:
            traceback.print_exc()
            logger.critical("An unexpected error caused the consumer to stop.", exc_info=True)
        finally:
            if self.consumer:
                logger.info("Closing Kafka consumer in finally block.")
                self.consumer.close()


    @staticmethod
    def clean_data(data):
        """清洗数据，数据转换 (保持不变)"""
        clean_data = data.copy()
        clean_data["event_id"] = clean_data.pop("df_event_id")
        # ... 其他清洗逻辑
        return clean_data

    # --- 优化: 拆分 `update_qy_knowledge`，使其职责更清晰 ---
    @staticmethod
    def _build_onetouch_data(intent2param) -> List[Dict]:
        """1、把同步的意图数据-覆盖更新qa_qy_onetouch.csv"""
        rows = []
        ff = lambda x: x.strip('\n').strip()
        # intent2param = self.knowledge["intent2param"]
        for _intent_id, content in intent2param.items():
            pri_question = content['primaryName']
            sim_questions = content["vocab"]['similary']
            qid = str(content["vocab"]["knowledgeId"])
            rows.append(
                {
                    "answer_content_list": [], 
                    "id": qid,
                    "question_id": qid,
                    "question_content": ff(pri_question),
                    "question_type": 0,    #    -- 问题类型（0标准问题、1知识点）
                    "similar_question_list": [
                        {
                            "question_id": qid,
                            "similar_id": qid + f"_{i+1}",
                            "similar_question": q
                        }
                        for i, q in enumerate(sim_questions)
                    ],
                    "category_all_name": "",
                    "status": 1,    # 已启用
                    # "longEffective": 1,   # 长期有效
                    "valid_begin_time": "",
                    "valid_end_time": "", 
                    "source": "一触即达意图",    # 七鱼一触即达
                }
            )
        return rows

    @staticmethod
    def _build_slot_data(ent2vocab) -> Dict[str, List[str]]:
        """2、将枚举实体写入 slot.json (entityType=1)"""
        slot2vocab = {}
        for eid, val in ent2vocab.items():
            if val.get("entityType") == 1:  # 枚举实体
                for entity, ev_list in val.get('vocab', {}).items():
                    entity_key = f"{val.get('entityName', '')}-{entity}"
                    slot2vocab[entity_key] = ev_list + [entity]
        return slot2vocab

    @staticmethod   
    def _write_json_file(data: Any, path: Path):
        """通用 JSON 文件写入函数"""
        try:
            with codecs.open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(f"  Successfully wrote data to {path}")
        except IOError as e:
            logger.error(f"  Failed to write to file {path}: {e}")

    def _trigger_downstream_update(self):
        """3、在独立线程中异步触发下游服务更新"""
        def _update_task():
            print("faq向量更新......")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
            try:
                update_api_url: str = "http://127.0.0.1:8098/update_faq_onetouch",
                headers = {"Content-Type": "application/json"}
                response = requests.get(update_api_url, headers=headers, timeout=30)
                response.raise_for_status() # 如果状态码是 4xx 或 5xx，则抛出异常
                logger.info(f"Downstream update triggered successfully. Status: {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to trigger downstream update: {e}")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
            print("【faq向量更新】完成!\n")
        
        # 请求本地update接口，更新全局变量（faiss & BM25 & slot）
        thread = threading.Thread(target=_update_task)
        thread.start()
        # thread.join()   # 等待线程完成
        print("=======faiss更新完成，消息字典清空=======")

    def _update_qy_knowledge_4_faq(self):
        """
        执行知识部署的完整流程：
        1. 生成并写入一触即达文件。
        2. 生成并写入 slot 文件。
        3. 异步触发下游服务更新。
        """
        # 将一触即达中的【意图问题和实体】写入本地json文件
        #（重建用于faq召回和槽位提取）
        _qa_qy_onetouch = self._build_onetouch_data(self.knowledge["intent2param"])
        _slot =  self._build_slot_data(self.knowledge["ent2vocab"])
        self._write_json_file(_qa_qy_onetouch, paths.kafka.onetouch)
        self._write_json_file(_slot, paths.kafka.slot)
        logger.info(f"【意图问题和实体】已写入本地json → {paths.kafka.onetouch} | {paths.kafka.slot}")
        # 备份 knowledge 数据
        self._write_json_file(self.knowledge["intent2param"], paths.kafka.intent)
        self._write_json_file(self.knowledge["param2ent"], paths.kafka.param)
        self._write_json_file(self.knowledge["ent2vocab"], paths.kafka.entity)
        
        # 异步触发下游更新
        logger.info("开始触发下游更新....")
        self._trigger_downstream_update()


def manual_sync_knowledge_snapshot():
    """手动将本地七鱼知识快照 push to Redis
    （第一次部署/调试用）
    """
    # 删除快照（测试）
    redis_client = RedisUtilsSentinel(redis_config.__dict__)
    redis_client.delete(snapshot_key)
    logger.info(f"删除快照成功：{snapshot_key}")
    # 读取已消费/缓存数据
    with open(paths.kafka.intent, 'r', encoding='utf-8') as file:
        json_string = file.read()
        intent2param = json.loads(json_string)
    with open(paths.kafka.param, 'r', encoding='utf-8') as file:
        json_string = file.read()
        param2ent = json.loads(json_string)
    with open(paths.kafka.entity, 'r', encoding='utf-8') as file:
        json_string = file.read()
        one_entity_dict = json.loads(json_string)
    with open(paths.entities.sys, 'r', encoding='utf-8') as file:
        json_string = file.read()
        sys_entity_dict = json.loads(json_string)
        sys_entity_dict = {x['id']:{"entityName": x["entityName"],"entityType": 0,"vocab": {}} for x in sys_entity_dict}
    with open(paths.entities.algo, 'r', encoding='utf-8') as file:
        json_string = file.read()
        algo_entity_dict = json.loads(json_string)
        algo_entity_dict = {x['id']:{"entityName": x["entityName"],"entityType": 5,"vocab": {}} for x in algo_entity_dict}
    ent2vocab = {**one_entity_dict, **sys_entity_dict, **algo_entity_dict}  # 加入系统实体和算法实体 → entity.json
    
    knowledges = {
        "intent2param": intent2param,
        "param2ent": param2ent,
        "ent2vocab": ent2vocab,
    }

    # 获取当前主题{0}的最新偏移量
    _consumer = KPConsumer(
        bootstrap_servers=kafka_config.bootstrap_servers,
        group_id=None,
        client_id=None,
        enable_auto_commit=False,
    )
    tp = TopicPartition(topic=kafka_config.topic_name, partition=0)
    # 获取最早和最新偏移量
    earliest_offsets = _consumer.beginning_offsets([tp])[tp]
    latest_offsets = _consumer.end_offsets([tp])[tp]
    # print(f"Topic '{kafka_config.topic_name}' 的 earliest_offsets: {earliest_offsets}")  # 2567
    # print(f"Topic '{kafka_config.topic_name}' 的 latest_offsets: {latest_offsets}")      # 148770
    _consumer.close()

    # 手动同步快照
    snap = {"data": knowledges, "offsets": {'0':latest_offsets}, "timestamp": time.time()}
    redis_client.set(snapshot_key, json.dumps(snap))
    logger.info("手动读取本地的七鱼知识快照，并保存到 Redis 👌")

    # 生成qa_qy_onetouch.json 和 slot.json
    _qa_qy_onetouch = KnowledgeSyncConsumer._build_onetouch_data(knowledges["intent2param"])
    _slot =  KnowledgeSyncConsumer._build_slot_data(knowledges["ent2vocab"])
    KnowledgeSyncConsumer._write_json_file(_qa_qy_onetouch, paths.kafka.onetouch)
    KnowledgeSyncConsumer._write_json_file(_slot, paths.kafka.slot)


def main(kafka_config, redis_config):
    try:
        # Redis 多机共享快照
        redis_client = RedisUtilsSentinel(redis_config.__dict__)
        logger.info("Successfully connected to Redis.")
    except Exception as e:
        logger.error(f"Could not connect to Redis: {e}. Snapshots will not be available.")
        redis_client = None
    
    # 创建并运行消费者
    kfk_consumer = KnowledgeSyncConsumer(
        kafka_config=kafka_config,
        redis_conn=redis_client,
    )
    kfk_consumer.run()


if __name__ == "__main__":

    # # 删除快照（测试）
    # redis_client = RedisUtilsSentinel(redis_config.__dict__)
    # redis_client.delete(snapshot_key)
    # logger.info(f"删除快照成功：{snapshot_key}")
    # exit()

    # # 七鱼知识快照 → push to Redis（第一次部署/调试用）
    # manual_sync_knowledge_snapshot()
    # exit()

    # 运行消费者
    main(kafka_config, redis_config)
