#!/bin/bash

# ================================== #
# # 线上推理准备工作
# ================================== #

# # 具体参考
# https://li.feishu.cn/wiki/FY2kwG73eisCFlkq6ozcDnmQnZc?from=from_copylink

# # 1、上传 model 至模型集

# # 2、申请推理服务，并挂载模型集

# 注意事项：
# 2.1 运行命令填入 python3 -m http.server 9009
# 2.2 端口映射 8098
# 2.3 环境变量设置

export FAQ_ENV=prod     # 线上生产环境
export CUDA_VISIBLE_DEVICES=0

# ================================== #
# # FAQ服务环境搭建
# # 仅在第一次部署 or 服务迁移时使用！
# ================================== #

# 安装必要的软件
apt-get update
apt-get install lsof

# 如果是第一次部署服务 or 服务迁移，clone 最新项目代码
git clone https://xxx/yyy.git

mkdir data_factory/kafka
mkdir data_factory/faq

# 手动 push 数据 → Redis
python -c "from api.data_sync.kafka_consumer import manual_sync_knowledge_snapshot; manual_sync_knowledge_snapshot()"
# 手动获取最新知识
python -c "from api.data_sync.get_crm_knowledge_v2 import *"

# 依赖安装(如果出问题，请先执行上一个命令)
pip install -r requirements.txt         # 针对 cu121 环境
pip install -r requirements-dev.txt       # 针对 cu117 环境
# cu121 还需额外安装依赖包
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install flash-attn --upgrade    # 2.7.4.post1


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
gunicorn application:app -c configs/gunicorn_config_api.py

# kill掉旧进程
pkill -f "socket8092_detection.py"
pkill -f consumer_qy_v2.py
pkill -f crontab_full_update_v2.py
pkill -f crontab_incremental_update_v2.py
pkill -f "gunicorn application:app -c configs/gunicorn_config_api.py"

# 1、启动主服务
nohup gunicorn application:app -c configs/gunicorn_config_api.py > logs/app.log 2>&1 &
lsof -i:8092
tail -f logs/app.log 
# 等待服务启动完毕...（Application startup complete.）

# 2、逐一启动辅助脚本

# 启动【服务端口监听】脚本
nohup python -u socket8092_detection.py > logs/socket.log 2>&1 &
ps -ef|grep socket

cd data_sync
cd data_sync

# 启动【知识拉取】定时任务脚本
nohup python -u crontab_full_update_v2.py > ../logs/full_update.log 2>&1 &
tail -f ../logs/full_update.log 
# 等待执行完毕...（【faq/recommend向量更新】完成!）
nohup python -u crontab_incremental_update_v2.py > ../logs/incremental_update.log 2>&1 &
tail -f ../logs/incremental_update.log
# 等待执行完毕...（增量「知识」更新完成）

ps -ef|grep crontab

# # 问题预警追踪(忽略)
# nohup python -u match_warning.py > ../logs/match_warning.log 2>&1 &

# 启动【kafka消息监听】服务
nohup python -u consumer_qy_v2.py > ../logs/consumer.log 2>&1 &
ps -ef|grep consumer
