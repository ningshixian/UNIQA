# UniFAQ
本项目参考自[haystack](https://github.com/deepset-ai/haystack)
技术文档 https://www.yuque.com/ningshixian/xa7g6q/edtx6bhb1vrafup5?singleDoc#
> 输入query文本 -> clean -> 召回（Recall） -> 粗序（Rank） -> 后处理（Rule） -> result

各模块详细介绍

📂 `uniqa/`  
│── 📂 `api/` → 
│    ├── `api.py` → 
│    ├── `router.py` → 
│── 📂 `configs/` → 
│    ├── `config.py` → Configurazioni globali (modelli, top_k, path, etc.)  
│    ├── `gunicorn_config_api.py` → 
│── 📂 `compoments/` → 各种具体的组件实现
│    ├──📂 `builders/` → 负责构建各种提示和答案，帮助用户更方便地与大语言模型（LLM）进行交互
│    │    ├── `answer_builder.py` → 其功能是根据模型的输出和相关上下文，通过正则匹配出答案。
│    │    ├── `chat_prompt_builder.py` → 其功能是用于构建聊天场景下的提示信息。
│    │    ├── `prompt_builder.py` → 其功能是用于构建通用的提示信息。
│    ├──📂 `web_search/` → 此模块提供了与网络搜索相关的组件
│    ├──📂 `converters/` → 负责文档的解析，确保文档格式适合模型处理
│    ├──📂 `embedders/` → 主要提供了使用 SentenceTransformers 模型进行嵌入的具体实现
│    ├──📂 `extractors/` → 主要提供了 NER，并将标注结果存储在文档的元数据中
│    ├──📂 `generators/` → 包括本地部署模型、OpenAI方式API调用
│    ├──📂 `preprocessors/` → 负责文档的清洗、分割、转换为 Document 对象
│    ├──📂 `rankers/` → 
│    │    ├── `lost_in_the_middle.py` → 使得最相关的文档位于开头或结尾，最不相关的文档位于中间。
│    │    ├── `sentence_transformers_similarity.py` → 使用预训练的cross-encoder模型排序
│    │    ├── `transformers_similarity.py` → 同上
│    ├──📂 `retrievers/` → 
│    │    ├──📂 `indexs/` → 定义了FAISS/ANNOY/MILVUS 索引文件
│    │    ├── `filter_retriever.py` → 根据指定的过滤器从文档存储中检索文档
│    │    ├── `sentence_window_retriever.py` → 与现有的检索器（如 BM25 检索器或嵌入检索器）协同工作，获取候选的相邻文档
│    │    ├── `EmbeddingRetriever.py` → 使用基于关键词的 BM25 算法从内存文档存储中检索与查询最相似的文档。
│    │    ├── `BM25Retriever.py` → 使用嵌入模型计算文本相似度从内存文档存储中检索与查询最相似的文档。
│    │    ├── `HybridRetriever.py` → 混合检索
│    ├──📂 `writers/` → 将文档写入向量数据库（document_stores）
│    ├──📂 `readers/` → ExtractiveQA。基于 Transformers 的抽取式问答模块，从文档中定位并提取与问题最匹配的文本片段
│── 📂 `core/` → 
│    ├──📂 `component/` → 定义了组件的基类和相关接口
│    ├── `errors.py` → 自定义错误
│    ├── `serialization.py` → 提供组件（Component）的序列化和反序列化功能
│── 📂 `dataclass/` → 定义了框架中使用的数据类，用于表示各种数据结构。
│    ├── `answer.py` → 答案模板，包括ExtractedAnswer、GeneratedAnswer
│    ├── `chat_message.py` → 对话模板，包括ChatMessage
│    ├── `document.py` → 定义了 Document 基本的数据类 ❗️
│    ├── `sparse_embedding.py` → 用于表示文档的稀疏嵌入向量
│    ├── `byte_stream.py` → 可用于处理文档中的二进制数据，像图片、音频等。
│── 📂 `data/` → 包含原始数据集和预处理数据集 
│── 📂 `document_stores/` → 负责存储和管理文档，为检索器提供数据支持
│    ├──📂 `types/` → 为文档存储的实现提供了统一的接口和规范。
│    ├── `document_store.py` → 实现了内存中的文档存储（写入、过滤、删除），提供 BM25 以及 向量余弦相似度检索
│    ├── `milvus_document_store.py` → 实现了基于 milvus 向量库的文档存储
│── 📂 `evaluation/` → 
│── 📂 `tools/` → 将组件包装为可调用的工具
│    ├── `vecs_whitening.py` → 一种处理向量空间坍缩的有效方法，非必须
│    ├── `socket_detection.py` → 
│── 📂 `utils/` → 通用工具
│── 📂 `training/` → embedding 训练
│── 📂 `logs/` → Contiene log per il debugging  
📂 `test/` → 测试目录，包含单元测试、集成测试等代码
📂 `examples/` → 示例代码目录
📂 `notebooks/` → 包含用于分析和实验的 Jupyter Notebook
📄 `requirements.txt` → Librerie necessarie  
📄 `README.md` → Documentazione del progetto  

---

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


## 其他资料
- https://github.com/iseesaw/FAQ-Semantic-Retrieval
- https://github.com/RUC-NLPIR/FlashRAG

