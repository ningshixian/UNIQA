# UniFAQ
一种 FAQ 混合检索解决方案 UniFAQ，你也可以称之为 SEARCH-U
- Semantic
- Enhanced
- Answer
- Retrieval
- CHatbot
- Unified

各模块详细介绍

📂 `hackathon_project/`  
│── 📂 `data/` → 包含原始数据集和预处理数据集 
│    ├── `database_piatti_con_id.csv`  
│    ├── `dish_mapping.json`  
│    ├── `domande.json`  
│    ├── `submission.csv`  
│    ├── 📂 `preprocessed/` → 已清理和优化的数据  
│── 📂 `models/` → 
│    ├── `embedding_model/` → embedding模型 (`bge-large`, `mpnet-base-v2`)  
│    ├── `reranker_model/` → reranking模型 (`cross-encoder/ms-marco-MiniLM-L-12-v2`)  
│    ├── `faiss_index/` → 预训练的 FAISS 索引文件
│── 📂 `src/` → Contiene il codice della pipeline  
│    ├── `preprocessing.py` → 数据清理与准备
│    ├── `retrieval.py` → FAISS + BM25 + TF-IDF  
│    ├── `reranking.py` → LLM排序 
│    ├── `generate_submission.py` → Pipeline completa per la submission  
│    ├── `config.py` → Configurazioni globali (modelli, top_k, path, etc.)  
│── 📂 `notebooks/` → 包含用于分析和实验的 Jupyter Notebook
│── 📂 `logs/` → Contiene log per il debugging  
│── 📂 `submission/` → Cartella con il file `submission.csv`  
│── 📄 `requirements.txt` → Librerie necessarie  
│── 📄 `README.md` → Documentazione del progetto  


📦Tecnologie Utilizzate
 **FAISS** → Per la ricerca veloce basata su similarità semantica.  
 **BM25** → Per il retrieval basato su parole chiave.  
 **TF-IDF** → Per il miglioramento della ricerca tra documenti simili.  
 **Sentence Transformers** → Per generare embedding NLP avanzati.  
 **Cross-Encoder LLM** → Per il reranking basato su IA.  
 **Pandas, Scikit-learn, NumPy** → Per la gestione e analisi dei dati.  

vecs_whitening，一种处理向量空间坍缩的有效方法，非必须，如果需要，可见本项目vecs_whitening.py代码，用法和sklearn的pca一致。可以将训练好的vecs_whitening模型地址输入bert_encoder中，也可以自己用本代码训练模型保存，再传入bert_encoder中。

---

## 项目介绍
> 输入query文本 -> clean -> 召回（Recall） -> 粗序（Rank） -> 后处理（Rule） -> result
> 技术文档 https://li.feishu.cn/wiki/S6p5w3gQ3i98PxkcGKNcicykned?fromScene=spaceOverview

- 问题理解，对用户 query 进行改写以及向量表示
- 召回模块，在问题集上进行候选问题召回，获得 topk（基于关键字的倒排索引 vs 基于向量的语义召回）
- 排序模块，对 topk 进行精排序

### 数据集

- 文本相似度数据集：百度千言项目发布了[文本相似度评测](https://aistudio.baidu.com/competition/detail/45/0/datasets)，包含 LCQMC/BQ Corpus/PAWS-X 等数据集（LCQMC百度知道问题匹配数据集、BQ微众银行智能客服问句匹配、PAWSX翻译成中文）
- FAQ知识库数据集:内部提供了一个 demo 版FAQ数据集，格式处理成下面json：

```
{
  "id": "001",
  "standard_question": "如何修改密码？",
  "similar_questions": [
    "怎么更改密码？",
    "密码怎么修改？",
    "我想改密码",
    "在哪里可以修改密码？"
  ],
  "answer": "您可以通过以下步骤修改密码：1.登录账户 2.进入个人中心 3.点击安全设置 4.选择修改密码",
  "category": "账户管理"
}
```

### 负采样
- 基于 Sklearn Kmeans 聚类, 在每个 query 所在聚类簇中进行负采样

### 微调 Embedding
fine-tune 过程主要进行文本相似度计算任务，亦句对分类任务；此处是为获得更好的句向量，基于Sentence-Transformers + CoSENT 进行训练.
Ranking loss 介绍：....

### FAQ Web服务

- Web 框架选择
    - 🔥 FastAPI + uvicorn（崩溃自动重启），最快的Python Web框架（实测的确比 Flask 快几倍）
- cache 缓存机制（保存最近的query对应的topic，命中后直接返回）
    - 🔥 functools.lru_cache() （默认缓存128，lru策略），装饰器，缓存函数输入和输出
- Locust 压力测试
    - 使用 Locust 编写压力测试脚本


## 使用说明

依赖安装
```
git clone https://github.com/ningshixian/Knowledge-QA-Assistant.git
pip install -r requirements.txt
```

负采样
```
python sampling.py \
	--filename='faq/train_faq.json' \
	--model_name_or_path='./model/bert-base-chinese' \
	--is_transformers=True \
	--hyper_beta=2 \
	--num_pos=5 \
	--local_num_negs=3 \
	--global_num_negs=2 \
	--output_dir='./samples'
```

embedding 有监督微调
```
cd faq-semantic-retrieval/module/lm
sh embedding_run.sh
```

直接测试FAQ效果
```
$ python faq.py
```

部署FAQ问答API服务
- Uvicorn 为单进程的 ASGI server
```
uvicorn router:app --host=0.0.0.0 --port=8091 --workers=1
```
- 而 Gunicorn 是管理运行多个 Uvicorn ，以达到并发与并行的最好效果。
```
gunicorn router:app -b 0.0.0.0:8098 -w 1 -t 50 -k uvicorn.workers.UvicornWorker
nohup gunicorn router:app -c configs/gunicorn_config_api.py > logs/router.log 2>&1 &
lsof -i:8098
```

Web服务压测
```
locust  -f locust_test.py  --host=http://127.0.0.1:8889/module --headless -u 100 -r 10 -t 3m
```

其他辅助脚本
- 服务监听&自动重启：`nohup python -u socket8098_detection.py > ../logs/socket.log 2>&1 &`
<!-- - 定时知识拉取：`nohup python -u crontab_data.py > ../logs/crontab.log 2>&1 &`
- 实时知识更新：`nohup python -u crontab_update_faq_know.py > ../logs/crontab_update_faq_know.log 2>&1 &`
- 问题预警追踪：`nohup python -u match_warning.py > ../logs/match_warning.log 2>&1 &`
- 七鱼一触即达kafka消息监听: `nohup python -u consumer_qy.py > ../logs/consumer.log 2>&1 &` -->


## 参考
- https://github.com/iseesaw/FAQ-Semantic-Retrieval
- https://github.com/RUC-NLPIR/FlashRAG

