# encoding:utf-8
import json
import time

from kafka import KafkaProducer, KafkaConsumer


class kafka_service(Object):
    # KAFKA_HOSTS = ["10.28.5.87:9092", "10.28.5.88:9092", "10.28.5.89:9092"]

    def __init__(self, kafka_ip_list):
        self.kafka_ip_list = kafka_ip_list

    def _producer(self, msg_dict, topic):
        producer = KafkaProducer(
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=5,  # 重试次数
            bootstrap_servers=self.kafka_ip_list,
            # # 传输时的压缩格式
            # compression_type="gzip",
        )
        producer.send(topic, value=msg_dict, partition=0)   # 发送到指定的消息主题（异步，不阻塞）
        producer.close()

    def _consumer(self, topic):
        consumer = KafkaConsumer(
            topic,
            group_id="test_group",
            auto_offset_reset="latest",
            enable_auto_commit=False,
            max_poll_records=1,
            bootstrap_servers=self.kafka_ip_list,
            session_timeout_ms=120000,
            request_timeout_ms=120001,
        )
        msg = next(consumer)
        return consumer, json.loads(msg.value)


