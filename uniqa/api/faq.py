# https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/33_Hybrid_Retrieval.ipynb#scrollTo=mSUiizGNytwX

import os
import sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('.'))

import re
import json
import copy
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import numpy as np

from uniqa import Document
from uniqa.components.preprocessors import DocumentCleaner, TextCleaner
from uniqa.components.preprocessors import RecursiveDocumentSplitter, ChineseDocumentSpliter
# from uniqa.components.joiners import DocumentJoiner
from uniqa.components.converters import PyPDFToDocument, JSONConverter
from uniqa.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from uniqa.components.writers import DocumentWriter
from uniqa.components.rankers import SentenceTransformersSimilarityRanker
from uniqa.components.rankers import DocumentJoiner
from uniqa.utils import ComponentDevice, VecsWhitening

from uniqa.document_stores.in_memory import InMemoryDocumentStore
from uniqa.components.retrievers.in_memory import InMemoryEmbeddingRetriever, InMemoryBM25Retriever

from uniqa.document_stores.milvus import MilvusDocumentStore
from uniqa.document_stores.milvus.function import BM25BuiltInFunction
from uniqa.components.retrievers.milvus import MilvusEmbeddingRetriever, MilvusSparseEmbeddingRetriever
from uniqa.components.retrievers.milvus import MilvusHybridRetriever


DEFAULT_CONNECTION_ARGS = {
    # "uri": "http://localhost:19530",  # 适用于 Milvus Docker 服务
    "uri": "./milvus_test.db",  # 本地 Milvus Lite
}


class DataPreprocessor:
    def __init__(self):
        # 初始化Cleaner(只处理 content)
        self.cleaner = DocumentCleaner(
            remove_empty_lines=True,
            remove_extra_whitespaces=False,
            remove_repeated_substrings=False,
            keep_id=True,   # keeps the IDs of the original documents.
            # remove_regex=r"\s\s+", 
            # remove_substrings = ["substring to remove"]
        )
    
    def _convert_qa_to_document(self, qa_item: Dict[str, Any]) -> List[Document]:
        """将QA结构体转换为Haystack Document（保留多条件答案）"""
        documents = []
        main_question = qa_item.get("questionContent", "")
        # 提取问题元数据（分类、状态等）
        question_meta = {
            "question_id": qa_item.get("questionId"),
            "question_type": qa_item.get("questionType"),
            # "long_effective": qa_item.get("longEffective", 0),  # 已删除字段（通过valid_begin_time、valid_end_time判断）
            "status": qa_item.get("status", 0),
            "category": qa_item.get("categoryAllName", ""),
            "valid_begin_time": qa_item.get("validStartTime"),
            "valid_end_time": qa_item.get("validEndTime"), 
        }
        # 存储所有带条件的答案(使用原始字段)
        answer_meta = qa_item.get("answerContentList", [])

        # 处理主问题
        if main_question:
            doc = Document(
                id=qa_item.get("questionId"),  # 主键id ==> primary_question_id
                content=main_question,  # Document内容为问题文本（用于语义检索）
                meta={
                    **question_meta,
                    "answers": answer_meta,  
                    "is_main_question": True # 添加一个标记，方便识别
                }
            )
            documents.append(doc)

        # 处理相似问题（每个相似问题作为独立Document，但共享同一套答案）
        for similar in qa_item.get("similarQuestionList", []):
            similar_question = similar.get("similarQuestion", "")
            if similar_question:
                doc = Document(
                    id=similar.get("similarId"),  # 主键id，避免重复
                    content=similar_question,
                    meta={
                        **question_meta,     # 用于关联回主问题ID
                        "answers": answer_meta,  # 复用主问题的答案列表
                        "similar_id": similar.get("similarId"),
                        "is_main_question": False
                    }
                )
                documents.append(doc)
        return documents

    def load_data(self, data_path):
        """
        将 QA JSON 结构转换为 Haystack Document 对象列表。
        返回值: List[Document]
        """
        with open(data_path, "r", encoding="utf-8") as file:
            self.data = json.load(file)  # List[Dict]

        docs = []
        for qa_item in self.data:
            _documents = self._convert_qa_to_document(qa_item)
            docs.extend(_documents)
        docs = self.cleaner.run(documents=docs)["documents"]
        return docs


class FAQPipeline:
    def __init__(self, is_whitening=False):
        """ FAQ pipeline

        Args:
            top_k (int, optional): 返回的结果数量. Defaults to 5.
            search_strategy (str, optional): 检索策略(hybrid / embedding / sparse). Defaults to 'hybrid'.
            is_whitening (bool, optional): 是否进行白话操作. Defaults to False.
        """
        # self.top_k = top_k
        # self.search_strategy = search_strategy
        self.is_whitening = is_whitening

        # 初始化白化模型（如果启用,需保证数据量足够大）
        if self.is_whitening:
            self.whitening_model = VecsWhitening(n_components=128)
        else:
            self.whitening_model = None
        
        # 初始化文本清理器
        self.cleaner = TextCleaner(remove_punctuation=True)

        # 初始化文档连接器
        self.doc_concat_joiner = DocumentJoiner(join_mode="concatenate", sort_by_score=True)

        # 初始化嵌入器
        self.doc_embedder = SentenceTransformersDocumentEmbedder(
            model="infgrad/stella-base-zh-v3-1792d",
            # meta_fields_to_embed=["similar_questions"],   # 选中的元数据会拼接到文档内容中进行嵌入
            # normalize_embeddings=True,  # 向量归一化
        )
        self.text_embedder = SentenceTransformersTextEmbedder(model="infgrad/stella-base-zh-v3-1792d")

        # 初始化排序器（可设置为全局变量）
        self.ranker = SentenceTransformersSimilarityRanker(
            model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            scale_score=True,
            # device=ComponentDevice.from_str("cpu"),  #
            # device=ComponentDevice.resolve_device(None),
        )

        # 预热
        self.doc_embedder.warm_up()
        self.text_embedder.warm_up()
        self.ranker.warm_up()

        # 初始化加权排序器
        from pymilvus import RRFRanker, WeightedRanker
        self.weighted_ranker = WeightedRanker(0.5, 0.5)  # (dense, sparse)

        # 占位
        self.milvus_document_store = None
        self.milvus_dense_retriever = None
        self.milvus_sparse_retriever = None
        self.milvus_hybrid_retriever = None 

    def setup_milvusDB_retriever(self, docs, filters=None):
        """
        1. 初始化 Milvus 文档存储 
        2. 将文档加载到 Milvus 向量数据库中，并为其构建索引，
            - 在加载过程中，可以选择性地对 Embedding 进行白化处理以提升检索效果。
        3. 初始化三种检索器（稠密、稀疏、混合），为后续的问答系统提供支持。

        :param docs: A list of documents to be loaded into the Milvus vector database.
        :param filters: A dictionary with filters to narrow down the search space (default is None).
        """
        # 配置BM25分析器参数（用于生成稀疏向量）
        analyzer_params_custom = {
            "tokenizer": "jieba",  # 使用结巴分词器处理中文
            "filter": [
                "lowercase",  # Built-in filter
                {"type": "stop", "stop_words": ["的", "了", "是"]},
            ],
            # "type": "chinese",    # 将chinese分析器套用到栏位,不接受任何可选参数
        }

        # 初始化 Milvus 文档存储
        self.milvus_document_store = MilvusDocumentStore(
            connection_args=DEFAULT_CONNECTION_ARGS,
            consistency_level="Session",  # Options: Strong, Bounded, Eventually, Session, Customized.
            drop_old=True,      # Drop the old collection if it exists and recreate it. ==》 DEFAULT_CONNECTION_ARGS
            index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}},
            # index_params={"index_type": "HNSW", "metric_type": "L2", "params": {"M": 16, "efConstruction": 64}},
            sparse_vector_field="sparse",
            sparse_index_params={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25", "params": {}},  # local mode only support SPARSE_INVERTED_INDEX
            builtin_function=[
                BM25BuiltInFunction(
                    function_name="bm25_function",
                    input_field_names="text",
                    output_field_names="sparse",  # same as sparse_vector_field
                    # You can customize the analyzer_params and enable_match here.
                    # See https://milvus.io/docs/analyzer-overview.md for more details.
                    analyzer_params=analyzer_params_custom,
                    enable_match=True,
                )
            ],
        )
        # 写入文档数据
        docs_with_embeddings = self.doc_embedder.run(docs)["documents"]   # meta embedding
        if self.is_whitening:  # 对 Embedding 执行白化操作
            self.whitening_model.fit(np.array([doc.embedding for doc in docs_with_embeddings]))
            docs_with_embeddings = [
                Document(
                    id=doc.id,
                    content=doc.content,
                    embedding=self.whitening_model.transform(np.array([doc.embedding])).squeeze(),
                    meta=doc.meta,
                )
                for doc in docs_with_embeddings
            ]
        num_docs = self.milvus_document_store.write_documents(docs_with_embeddings)  # return int
        print(f"DocumentStore 加载完成：{num_docs}个问题文档，包含多条件答案")
        
        # Initializes a new instance of the MilvusXXXRetriever.
        self.milvus_dense_retriever = MilvusEmbeddingRetriever(document_store=self.milvus_document_store, filters=filters)
        self.milvus_sparse_retriever = MilvusSparseEmbeddingRetriever(document_store=self.milvus_document_store, filters=filters)
        self.milvus_hybrid_retriever = MilvusHybridRetriever(
            document_store=self.milvus_document_store,
            reranker=self.weighted_ranker,
            filters=filters
        )

    def _join_deduplicate(
        self,
        candidate_docs_list: List[List[Document]],
    ):
        """对多个列表中的 Document 对象进行合并与去重操作"""
        documents = []
        for candidate_docs in candidate_docs_list:
            docs = copy.deepcopy(candidate_docs)
            for doc in docs:
                doc.id=doc.meta["question_id"]
            documents.append(docs)
        # 策略合并: 去重+保留最大得分的 document.id
        results = self.doc_concat_joiner.run(documents=documents)["documents"]
        return results

    def run(self, query, top_k=5, search_strategy='hybrid'):
        """根据输入的查询语句，从 Milvus 中检索相关的文档。

        Args:
            query (str): 需要检索的问题文本。
            top_k (int, optional): 返回的最相关结果的数量，默认为 5。
            search_strategy (str, optional): 检索策略，支持以下三种模式：
                - 'embedding': 使用稠密向量进行语义检索；
                - 'sparse': 使用稀疏向量（BM25）进行关键词匹配；
                - 'hybrid': 混合使用稠密向量和稀疏向量进行检索，默认值。
        
        Returns:
            List[Document]: 包含检索结果的文档列表
        """
        if not self.milvus_document_store:
            raise ValueError("MilvusDocumentStore is not initialized. Call `load_milvus()` first.")

        # 获取问题向量
        query = self.cleaner.run([query])["texts"][0]
        query_embedding = self.text_embedder.run(query)["embedding"]
        if self.is_whitening:
            query_embedding = self.whitening_model.transform(np.array([query_embedding])).squeeze()
        
        # 知识检索
        if search_strategy == 'embedding':  # 语义检索
            self.milvus_dense_retriever.top_k = top_k * 6
            candidate_docs = self.milvus_dense_retriever.run(query_embedding=query_embedding)["documents"]
            candidate_docs = self._join_deduplicate([candidate_docs])   # 删除重复的文档
        elif search_strategy == 'sparse':   # bm25检索
            self.milvus_sparse_retriever.top_k = top_k * 6
            candidate_docs = self.milvus_sparse_retriever.run(query_text=query)["documents"]
            candidate_docs = self._join_deduplicate([candidate_docs])   # 删除重复的文档
        elif search_strategy == 'hybrid':
            # candidate_docs = self.milvus_hybrid_retriever.run(
            #     query_embedding=query_embedding,
            #     query_text=query, 
            # )["documents"]
            self.milvus_dense_retriever.top_k = top_k * 3
            self.milvus_sparse_retriever.top_k = top_k * 3
            semantic_hit = self.milvus_dense_retriever.run(query_embedding=query_embedding)["documents"]
            sparse_hit = self.milvus_sparse_retriever.run(query_text=query)["documents"]

            join_hits = self._join_deduplicate([semantic_hit, sparse_hit])   # 删除重复的文档

            # 重排序(给的 score 较低)
            rank_hits = self.ranker.run(query=query, documents=join_hits, top_k=top_k * 6)["documents"]

            # 策略合并: 线性加权融合
            doc_joiner = DocumentJoiner(
                join_mode="merge",     # 融合方式(concatenate / merge / reciprocal_rank_fusion / distribution_based_rank_fusion)
                weights=[0.5, 0.5],    # (ranked, embedded) → [0.5, 0.5]
                # top_k=top_k,
                sort_by_score=True
            )
            candidate_docs = doc_joiner.run(documents=[rank_hits, semantic_hit])["documents"]
        else:
            raise ValueError("Invalid mode")
        return candidate_docs

    def _match_conditions(self, answer: Dict[str, Any], user_conditions: Dict[str, Any]) -> float:
        """计算答案与用户条件的匹配度（0~1），用于 Metadata Filtering 无法处理的复杂筛选逻辑。
        answer新增is_default_answer字段。如果一个答案没有(车型标签、生效时间、最高/最低ota版本)，就认为是默认答案。

        Args:
            answer (Dict[str, Any]): 包含答案信息的字典，====》candidate_docs[*].meta["answers"][*]
            user_conditions (Dict[str, Any]): 用户提供的检索条件，例如 {"car_type": ["Model S 高端系列"], "ota_version": ["7.0"]}

        Returns:
            float: 表示匹配度的分数，范围在 0 到 1 之间。分数越高表示匹配度越高。
                - 车型匹配：若用户指定的车型存在于答案中，加分权重最高（+0.4）。
                - OTA版本匹配：若用户提供的版本在答案指定的版本范围内，加分权重次之（+0.3）。
                - 其他条件可扩展（如用户类型、优先级等）。
        """
        score = 0.0

        # 1. 车型匹配
        # 如果用户提供了车辆信息 (user_cars 不为空)，并且其中至少有一种车型包含在答案 car_label 中，则增加评分。
        # 如果用户没有提供车辆信息 (user_cars 为空)，并且答案 car_label 也为空，则默认增加评分。
        car_label = " / ".join([x["vehicleModelName"]+" "+x["vehicleSeriesName"] for x in answer.get("carLabelList", [])])
        user_cars = user_conditions.get("car_type", [])
        _func = lambda x: re.sub(r'[_\s]', '', x.lower())   # 去除‘车型’里的空格和下划线
        if user_cars:
            if any([_func(car) in _func(car_label) for car in user_cars]):
                score += 0.4
        else:
            if not car_label:
                score += 0.4

        # 2. OTA版本匹配（检查是否在[lowest, highest]范围内）
        # 如果用户提供了 OTA 版本号，只要其中一个 ota 满足版本范围要求，则加分 0.3。
        # 如果用户未提供 OTA 版本号，并且答案也没有设置任何版本范围限制；默认加分 0.3，表示无版本约束时视为匹配。
        lowest_v = answer.get("lowestOtaVersion", "")
        # lowest_v = lowest_v if lowest_v else "0.0"
        highest_v = answer.get("highestOtaVersion", "")
        # highest_v = highest_v if highest_v else "999.0"
        user_ota_versions = user_conditions.get("ota_version", [])
        if user_ota_versions:
            if lowest_v and highest_v and any([float(lowest_v) <= float(user_v) <= float(highest_v) for user_v in user_ota_versions]):
                score += 0.3
        else:
            if not lowest_v and not highest_v:
                score += 0.3

        # # 3. 用户类型匹配（存在于允许列表则加分）
        # user_types = answer_conditions.get("userTypes", [])
        # user_type = user_conditions.get("user_type")
        # if user_type and user_type in user_types:
        #     score += 0.2  # 用户类型权重较低

        # # 4. 优先级调整（答案自带优先级，高优先级加分）
        # priority = answer_conditions.get("priority", 0)
        # score += priority * 0.01  # 优先级微调（避免权重过高）

        return round(score, 3)


def pretty_print_results(prediction):
    for doc in prediction:
        print(doc.id, doc.content, doc.score)
        print(doc.meta)
        # print(doc.embedding)
        print("\n", "\n")


if __name__ == "__main__":

    # 数据获取和处理
    preprocessor = DataPreprocessor()
    docs = preprocessor.load_data(data_path="uniqa/data/demo.json")

    # Do Metadata Filtering
    filters={
        "operator": "AND",
        "conditions": [
            # {"field": "score", "operator": ">", "value": 0},    # 非meta数据❌
            # {"field": "meta.answes[*].status", "operator": "==", "value": 1},   # Milvus 不支持嵌套字段查询❌
            {"field": "meta.status", "operator": "==", "value": 1},  # ✓
            # 过滤过期问题 ✓
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.valid_begin_time", "operator": "==", "value": None},
                    {"field": "meta.valid_begin_time", "operator": "<=", "value": str(datetime.now())},
                ],
            }, 
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.valid_end_time", "operator": "==", "value": None},
                    {"field": "meta.valid_end_time", "operator": ">=", "value": str(datetime.now())},
                ],
            },
            # TEXT_MATCH
        ]
    }

    faq = FAQPipeline()
    faq.setup_milvusDB_retriever(docs, filters)

    top_k=10
    search_strategy='embedding'

    while True:
        query = input("请输入问题：")   # 如何更新OTA系统 ｜ 怎样升级车载系统 ｜ 软件更新方法
        if query == "exit":
            break
        results = faq.run(query, top_k=top_k, search_strategy=search_strategy)
        
        # 复杂filters逻辑 → 筛选答案
        candidate_answers = []
        user_conditions = {"car_type": [], "ota_version": ["7.0"]}
        for doc in results:
            answers = doc.meta.get("answers", [])
            # 计算答案与用户条件的匹配度
            score_list = []
            for ans in answers:
                match_score = faq._match_conditions(ans, user_conditions)  # 条件匹配度（0~1）
                score_list.append(match_score)
            # 筛选 score_list中大于等于0.7 的index
            index_list = [i for i, score in enumerate(score_list) if score >= 0.7]
            if index_list:
                candidate_answers.append(
                    {
                        "question": doc.content,  # 匹配的问题
                        "question_id": doc.meta.get("question_id"),
                        "question_type": doc.meta.get("question_type"),
                        "category": doc.meta.get("category", ""), 
                        "is_main_question": doc.meta.get("is_main_question"),
                        "score": round(doc.score, 4),   # 匹配得分
                        "answer": [answers[i] for i in index_list]  # 符合条件的答案
                    }
                )
            else:
                # 获取默认答案索引（车型标签、生效时间、最高/最低ota版本都为空的元素）
                default_index = None
                for i, ans in enumerate(answers):
                    if not ans.get("car_type") and not ans.get("effective_time") and not ans.get("max_ota_version") and not ans.get("min_ota_version"):
                        default_index = i
                        break
                # 添加默认答案
                if default_index is not None and default_index not in index_list:
                    candidate_answers.append(
                    {
                        "question": doc.content,  # 匹配的问题
                        "question_id": doc.meta.get("question_id"),
                        "question_type": doc.meta.get("question_type"),
                        "category": doc.meta.get("category", ""), 
                        "is_main_question": doc.meta.get("is_main_question"),
                        "score": round(doc.score, 4),   # 匹配得分
                        "answer": [answers[default_index]]  # 默认答案
                    }
                )

        # 按综合得分排序，取Top K答案
        candidate_answers.sort(key=lambda x: x["score"], reverse=True)
        candidate_answers = candidate_answers[:top_k]
        print("Candidate Answers:", candidate_answers)
        # pretty_print_results(candidate_answers)

    """
    [
        {'question': '如何更新OTA系统？', 'question_id': 'KB001', 'question_type': 0, 'category': '基本操作/车辆控制/系统设置/软件更新', 'is_main_question': False, 'score': 0.9939, 
        'answer': [{
            'answerContent': '...', 
            'carLabelList': [{...}], 
            'highestOtaVersion': '7.1', 
            'lowestOtaVersion': '6.0', 
            ...,
            'validEndTime': '2035-05-15 16:09:40', 
            'validStartTime': '2025-05-15 16:09:40', 
            'status': 1
        },{}...,{}]
        }, 
    ]
    """
