import sys
import time
import pytz
from datetime import datetime
import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

import subprocess
from pytz import timezone


def get_now_time():
    # PRC为北京时间 CST为中国标准时间
    beijing_tz = pytz.timezone('PRC')
    now = datetime.now(beijing_tz).strftime("%Y-%m-%d %H:%M:%S")
    return now


def scheduled_job():
    print("获取【知识平台】QA数据对（通用&机器人），定时任务启动!")
    print(get_now_time())
    loader = subprocess.Popen(["python", "get_crm_knowledge_v2.py"])
    returncode = loader.wait()  # # 等待子进程结束，并获取退出状态码
    if returncode == 0:
        print("【知识拉取】成功！")
    else:
        print("【知识拉取】失败！")
    print(get_now_time())


def scheduled_job2():
    print("faq向量更新......")
    print(get_now_time())
    url = "http://127.0.0.1:8092/full_update"
    headers = {"Content-Type": "application/json"}
    response = requests.request("GET", url, headers=headers)
    if response.status_code == 200:
        print("【faq向量更新】成功！")
    else:
        print("【faq向量更新】失败！")
    print(get_now_time())


def dojob():
    # 创建调度器：BlockingScheduler 
    # 指定时区 timezone=pytz.timezone("Asia/Shanghai") 
    scheduler = BlockingScheduler(timezone=pytz.timezone('PRC'))

    # # 未显式指定，那么则立即执行
    # scheduler.add_job(auto_update_json, args=[])

    # # 添加定时任务，每5分钟执行一次
    # scheduler.add_job(scheduled_job, trigger=CronTrigger.from_crontab('*/5 * * * *'), id='knowledge accquire')
    # 添加定时任务，每天凌晨12点 trigger='cron' 
    scheduler.add_job(scheduled_job, trigger=CronTrigger.from_crontab('0 0 * * *'), id='knowledge accquire')
    scheduler.add_job(scheduled_job2, trigger=CronTrigger.from_crontab('5 0 * * *'), id='fqa update')
    scheduler.start()


if __name__ == "__main__":
    dojob()
