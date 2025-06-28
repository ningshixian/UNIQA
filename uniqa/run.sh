#!/bin/bash

# ================================== #
# # lpai 线上推理准备工作
# ================================== #

# # 具体参考 LPAI 推理服务部署
# https://li.feishu.cn/wiki/FY2kwG73eisCFlkq6ozcDnmQnZc?from=from_copylink

# # 1、上传 model 至模型集
# 由于是在 101 训练，需要手动下载至本地，或将模型复制到 lpai 机器的数据卷中（eg. nlp_data），再执行下面的命令
# 命令使用方法：lpa create models/{model_name}/versions/{model_version} -l 'source=sdk' -desc {model_describe} --format dir -paths 'pvc://{pvc_name}/{dir_path}/'
# 具体使用示例：lpa create models/faq-for-dialogue/versions/24-09-18-1 --format dir -ns enterpris-smartbusiness -paths 'pfs://prod-kubeflow/jfs-lpai01/pvc-40b89b1e-8847-427f-9930-4daf76d7eb28/nsx/models/stella-fine-tuned-v4.5/'

# # 2、申请推理服务，并挂载模型集
# https://lpai.lixiang.com/lpaiweb/xspacex_enterpris-smartbusiness/subApp/model/inference/detail/dialogue-faq-service1
# https://lpai.lixiang.com/lpaiweb/xspacex_enterpris-smartbusiness/subApp/model/inference/detail/dialogue-faq-service2

# 注意事项：
# 2.1 运行命令填入 python3 -m http.server 9009
# 2.2 端口映射 8098
# 2.3 环境变量设置 export FAQ_ENV=prod     # 线上生产环境


# ================================== #
# # FAQ服务环境搭建
# # 仅在第一次部署 or 服务迁移时使用！
# ================================== #

# # 安装必要的软件
# apt-get update
# apt-get install lsof

# # 如果是第一次部署服务 or 服务迁移，clone 最新项目代码
# git clone https://gitlab.chehejia.com/ningshixian/faq-semantic-retrieval.git

# # 七鱼一触即达数据手动拷贝&更新（仅在第一次部署 or 服务迁移时使用！）
# # 在旧机器/旧环境下，拷贝最新的一触即达数据并提交git
# cp -r faq-semantic-retrieval/data_factory/kafka_prod/* kafka_prod_bak/
# cd kafka_prod_bak & git push origin master
# # 在新机器/新环境下，下载最新线上的一触即达数据并覆盖更新
# git clone https://gitlab.chehejia.com/ningshixian/kafka_prod_bak.git
# cp -r kafka_prod_bak/* faq-semantic-retrieval/data_factory/kafka_prod/

# # 针对 cu121 环境
# 参照 requirements.txt 安装依赖包

# # 针对 cu117 环境
# 参照 requirements-dev.txt 安装依赖包

# # 部分代码需修改下（忽略！）
# vi configs/gunicorn_config_api.py 
# workers = 1   # 多个worker，性能会好些，但全局变量更新无法共享！
# timeout = 240  # 服务启动超时，默认为 240s

# vi router.py & vi faq.py
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 注释掉，否则会报错!


# ================================== #
# # FAQ服务启动/重启步骤
# ================================== #

# conda activate pp # 测试激活，线上无需

# 拉取最新项目代码
git pull
# 拉取最新项目代码(强制)
git fetch --all
git reset --hard origin/master

# 验证下（ok就直接退出）
gunicorn router:app -c configs/gunicorn_config_api.py

# kill掉旧进程
pkill -f "socket8098_detection.py"
pkill -f consumer_qy.py
pkill -f crontab_data.py
pkill -f crontab_update_faq_know.py
pkill -f "gunicorn router:app -c configs/gunicorn_config_api.py"

# 1、启动主服务
nohup gunicorn router:app -c configs/gunicorn_config_api.py > logs/router.log 2>&1 &
lsof -i:8098
tail -f logs/router.log 
# 等待服务启动完毕...（Application startup complete.）

# 2、逐一启动辅助脚本

# 启动【服务端口监听】脚本
nohup python -u socket8098_detection.py > logs/socket.log 2>&1 &
ps -ef|grep socket

cd data_sync
cd data_sync

# 启动【知识拉取】定时任务脚本
nohup python -u crontab_data.py > ../logs/crontab.log 2>&1 &
tail -f ../logs/crontab.log 
# 等待执行完毕...（【faq/recommend向量更新】完成!）
nohup python -u crontab_update_faq_know.py > ../logs/crontab_update_faq_know.log 2>&1 &
tail -f ../logs/crontab_update_faq_know.log
# 等待执行完毕...（增量「知识」更新完成）

ps -ef|grep crontab

# # 问题预警追踪(忽略)
# nohup python -u match_warning.py > ../logs/match_warning.log 2>&1 &

# 启动【kafka消息监听】服务
nohup python -u consumer_qy.py > ../logs/consumer.log 2>&1 &
ps -ef|grep consumer


# ================================== #
# # FAQ服务检查
# ================================== #

# cd ..

# 启动完毕后，检查日志是否正常
# vi logs/router.log
# vi logs/crontab.log
# vi logs/crontab_update_faq_know.log 
# vi logs/consumer.log
# 最后是看下数据是否正常同步过来了
# ll -ht data_factory/
# ll -ht data_factory/kafka_prod/
# vi data_factory/kafka_prod/slot.json
# vi data_factory/kafka_prod/qa_qy_onetouch.csv

# # 出问题的话回退版本
# git reset --hard HEAD~1   # 回退到上一个版本
# git reset --hard aaf505d5d55476843989826a0895a08356194b4c # 回退到指定版本
