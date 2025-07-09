# https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/33_Hybrid_Retrieval.ipynb#scrollTo=mSUiizGNytwX

import os
import sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('.'))

import json
from typing import List
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
    "uri": "./milvus_test.db",  # 适用于 Milvus Lite 本地
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
    
    def load_data(self):
        """
        将 QA JSON 结构转换为 Haystack Document 对象列表。
        """
        docs = [
            Document(
                id="KB001",
                content="如何注册新的用户账号？",   # standard_question
                meta={
                    "similar_questions": [
                        "怎样创建我的新账户？",
                        "注册流程是怎样的？",
                        "我该如何开始注册？",
                        "新用户注册步骤是什么？",
                    ],
                    "answer": [
                        {"content": "默认答案","car_type":None, "ota_version": None, "valid_begin_time": None, "valid_end_time": None}, 
                        {"content": "您可以通过访问我们的官方网站，点击页面右上角的“注册”按钮，然后按照提示填写您的邮箱、密码等信息来创建新账号。","car_type":"理想L8/Ultra智能焕新版,理想L8/2024款 Ultra", "ota_version": "7.0.2", "valid_begin_time": None, "valid_end_time": None}, 
                        {"content": "如果您已经注册了账号，您可以通过点击页面右上角的“登录”按钮，然后输入您的邮箱和密码来登录。","car_type":"理想L9/Ultra智能焕新版,理想L9/2024款 Ultra", "ota_version": "7.0.2", "valid_begin_time": datetime(2025, 1, 1).strftime('%Y-%m-%d %H:%M:%S'), "valid_end_time": datetime(2025, 7, 8).strftime('%Y-%m-%d %H:%M:%S')},
                    ], 
                    "category": "账户管理",
                    "valid_begin_time": "2025-05-15 16:09:40", 
                    "valid_end_time": "2035-05-15 16:09:40"
                },
            ),
            Document(
                id="KB002",
                content="忘记密码了怎么办？",    # standard_question
                meta={
                    "similar_questions": [
                        "我忘了我的登录密码，该如何找回？",
                        "密码丢失了，怎么重置？",
                        "如果我忘记了账户密码，有什么解决办法？",
                        "如何恢复我的账户密码？",
                    ],
                    "answer": [
                        {"content": "默认答案","car_type":None, "ota_version": None, "valid_begin_time": None, "valid_end_time": None}, 
                        {"content": "如果您忘记了密码，请点击登录页面的“忘记密码”链接，输入您的注册邮箱，我们会发送一封包含重置密码链接的邮件给您。","car_type":"理想L6/Ultra智能焕新版,理想L6/2024款 Ultra", "ota_version": "6.0.2", "valid_begin_time": None, "valid_end_time": None},
                        {"content": "如果您的账户被其他用户所使用，您可以在登录页面点击“忘记密码”链接，输入您的注册邮箱，我们会发送一封包含重置密码链接的邮件给您。", "car_type": "理想L6/Ultra智能焕新版,理想L6/2024款 Ultra", "ota_version": "6.0.2", "valid_begin_time": None, "valid_end_time": None}
                    ],
                    "category": "账户管理",
                    "valid_begin_time": "2025-05-15 16:09:40", 
                    "valid_end_time": "2035-05-15 16:09:40"
                },
            ),
        ]
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

        # 初始化嵌入器
        self.doc_embedder = SentenceTransformersDocumentEmbedder(
            model="infgrad/stella-base-zh-v3-1792d",
            # meta_fields_to_embed=["similar_questions"],   # 选中的元数据会拼接到文档内容中进行嵌入
            # normalize_embeddings=True,  # 向量归一化
        )
        self.text_embedder = SentenceTransformersTextEmbedder(model="infgrad/stella-base-zh-v3-1792d")

        # 初始化排序器
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
        self.filters = None     # for Metadata Filtering

        # filters={   # Do Metadata Filtering
        #     "operator": "AND",
        #     "conditions": [
        #         {"field": "meta.answer[0].ota_version", "operator": ">", "value": 1.21},
        #         {"field": "meta.valid_time", "operator": ">", "value": datetime(2023, 11, 7)},
        #     ],
        # },

    def load_milvus(self, docs):
        # 初始化 Milvus 文档存储
        self.milvus_document_store = MilvusDocumentStore(
            connection_args=DEFAULT_CONNECTION_ARGS,
            consistency_level="Session",  # Options: Strong, Bounded, Eventually, Session, Customized.
            drop_old=True,      # 是否删除旧集合 → DEFAULT_CONNECTION_ARGS
            index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}},
            # index_params={"index_type": "HNSW", "metric_type": "L2", "params": {"M": 16, "efConstruction": 64}},
            sparse_vector_field="sparse",
            sparse_index_params={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25", "params": {}},
            # local mode only support SPARSE_INVERTED_INDEX
            builtin_function=[
                BM25BuiltInFunction(
                    function_name="bm25_function",
                    input_field_names="text",
                    output_field_names="sparse",  # same as sparse_vector_field
                    # You can customize the analyzer_params and enable_match here.
                    # See https://milvus.io/docs/analyzer-overview.md for more details.
                    # analyzer_params=analyzer_params_custom,
                    # enable_match=True,
                )
            ],
        )

        # 文档写入 milvus_document_store
        docs_with_embeddings = self.doc_embedder.run(docs)["documents"]   # meta embedding

        if self.is_whitening:
            # 对 Embedding 执行白化操作
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

        # 写入 MilvusDocumentStore
        self.milvus_document_store.write_documents(docs_with_embeddings)  # return int
        
        # 加载检索模块
        self.milvus_dense_retriever = MilvusEmbeddingRetriever(document_store=self.milvus_document_store, filters=self.filters)
        self.milvus_sparse_retriever = MilvusSparseEmbeddingRetriever(document_store=self.milvus_document_store, filters=self.filters)
        self.milvus_hybrid_retriever = MilvusHybridRetriever(
            document_store=self.milvus_document_store,
            reranker=self.weighted_ranker,
            filters=self.filters
        )

    def run(self, query, top_k=5, search_strategy='hybrid'):
        if not self.milvus_document_store:
            raise ValueError("MilvusDocumentStore is not initialized. Call `load_milvus()` first.")

        # 获取问题向量
        query = self.cleaner.run([query])["texts"][0]
        query_embedding = self.text_embedder.run(query)["embedding"]
        if self.is_whitening:
            query_embedding = self.whitening_model.transform(np.array([query_embedding])).squeeze()
        
        if search_strategy == 'embedding':  # 语义检索
            self.milvus_dense_retriever.top_k = top_k
            candidate_docs = self.milvus_dense_retriever.run(query_embedding=query_embedding)["documents"]
        elif search_strategy == 'sparse':   # bm25检索
            self.milvus_sparse_retriever.top_k = top_k
            candidate_docs = self.milvus_sparse_retriever.run(query_text=query)["documents"]
        elif search_strategy == 'hybrid':
            # candidate_docs = self.milvus_hybrid_retriever.run(
            #     query_embedding=query_embedding,
            #     query_text=query, 
            # )["documents"]
            self.milvus_dense_retriever.top_k, self.milvus_sparse_retriever.top_k = top_k, top_k
            semantic_hit = self.milvus_dense_retriever.run(query_embedding=query_embedding)["documents"]
            sparse_hit = self.milvus_sparse_retriever.run(query_text=query)["documents"]

            # 策略合并: 拼接+去重
            doc_joiner = DocumentJoiner(
                join_mode="concatenate",     # 融合方式(concatenate / merge / reciprocal_rank_fusion / distribution_based_rank_fusion)
                top_k=top_k * 2,
                sort_by_score=True
            )
            candidate_docs = doc_joiner.run(documents=[semantic_hit, sparse_hit])["documents"]

            # 重排序(给的 score 较低)
            candidate_docs_rank = self.ranker.run(query=query, documents=candidate_docs, top_k=top_k * 2)["documents"]

            # 策略合并: 线性加权融合
            doc_joiner = DocumentJoiner(
                join_mode="merge",     # 融合方式(concatenate / merge / reciprocal_rank_fusion / distribution_based_rank_fusion)
                weights=[0.5, 0.5],    # (ranked, embedded) → [0.5, 0.5]
                top_k=top_k,
                sort_by_score=True
            )
            candidate_docs = doc_joiner.run(documents=[candidate_docs_rank, semantic_hit])["documents"]

        return candidate_docs


def pretty_print_results(prediction):
    for doc in prediction:
        print(doc.id, doc.content, doc.score)
        print(doc.mata)
        # print(doc.embedding)
        print("\n", "\n")


if __name__ == "__main__":

    # 数据获取和处理
    preprocessor = DataPreprocessor()
    docs = preprocessor.load_data()

    faq = FAQPipeline()
    faq.load_milvus(docs)

    while True:
        query = input("请输入问题：")   # "账号注册"
        if query == "exit":
            break
        prediction = faq.run(query, top_k=5, search_strategy='embedding')
        pretty_print_results(prediction)

    """
    Document(id=KB001, content: '如何注册新的用户账号？', meta: {
        'similar_questions': [...], 
        'answer': [{...}, {...}, {...}], 
        'category': '', 
        'valid_begin_time': , 
        'valid_end_time': 
    }, score: 0.9303537011146545, embedding: vector of size 1792)
    """
