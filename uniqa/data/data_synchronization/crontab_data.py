import sys
import time
import pytz
from datetime import datetime
import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

import subprocess
from pytz import timezone

"""
知识平台数据-全量同步
nohup python crontab_data.py > ../logs/crontab.log 2>&1 &
"""

def get_now_time():
    # PRC为北京时间 CST为中国标准时间
    beijing_tz = pytz.timezone('PRC')
    now = datetime.now(beijing_tz).strftime("%Y-%m-%d %H:%M:%S")
    return now


def scheduled_job():
    # 重启服务
    # loader = subprocess.Popen(["pkill", "-f", "crontab_retraining.py"])
    # returncode = loader.wait()  # 阻塞直至子进程完成
    print("获取【知识平台】QA数据对（通用&机器人），定时任务启动!")
    print(get_now_time())
    loader = subprocess.Popen(["python", "get_crm_knowledge.py"])
    returncode = loader.wait()  # 阻塞直至子进程完成
    print(get_now_time())
    print("【知识拉取】schedule完成!\n")

    print("获取【知识平台】车型数据，定时任务启动!")
    print(get_now_time())
    loader = subprocess.Popen(["python", "get_car_type.py"])
    returncode = loader.wait()  # 阻塞直至子进程完成
    print(get_now_time())
    print("【车型拉取】schedule完成!\n")

    print("对知识数据进行清洗和处理 --> qa4api --> qa4rec")
    print(get_now_time())
    loader = subprocess.Popen(["python", "knowledge_process_4_sync.py"])
    returncode = loader.wait()  # 阻塞直至子进程完成
    print(get_now_time())
    print("【qa知识库】更新完毕!\n")


def scheduled_job2():
    print("faq向量更新......")
    print(get_now_time())
    url = "http://127.0.0.1:8098/update_faq"
    headers = {"Content-Type": "application/json"}
    response = requests.request("GET", url, headers=headers)
    print(get_now_time())
    print("【faq向量更新】完成!\n")


def scheduled_job3():
    print("recommend向量更新......")
    print(get_now_time())
    url = "http://127.0.0.1:8098/update_recommend"
    headers = {"Content-Type": "application/json"}
    response = requests.request("GET", url, headers=headers)
    print(get_now_time())
    print("【recommend向量更新】完成!\n")


def auto_update_json():
    scheduled_job()
    scheduled_job2()
    scheduled_job3()


def dojob():
    # 创建调度器：BlockingScheduler 
    # 指定时区 timezone=pytz.timezone("Asia/Shanghai") 
    scheduler = BlockingScheduler(timezone=pytz.timezone('PRC'))

    # 未显式指定，那么则立即执行
    scheduler.add_job(auto_update_json, args=[])
    # # 添加定时任务，每5分钟执行一次
    # scheduler.add_job(scheduled_job, trigger=CronTrigger.from_crontab('*/5 * * * *'), id='knowledge accquire')
    # 添加定时任务，每天凌晨12点 trigger='cron' 
    scheduler.add_job(scheduled_job, trigger=CronTrigger.from_crontab('0 0 * * *'), id='knowledge accquire')
    scheduler.add_job(scheduled_job2, trigger=CronTrigger.from_crontab('5 0 * * *'), id='fqa update')
    scheduler.add_job(scheduled_job3, trigger=CronTrigger.from_crontab('7 0 * * *'), id='recommend update')
    scheduler.start()


if __name__ == "__main__":
    dojob()
