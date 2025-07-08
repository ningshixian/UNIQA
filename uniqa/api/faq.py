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
from uniqa.components.preprocessors import DocumentCleaner
from uniqa.components.preprocessors import RecursiveDocumentSplitter, ChineseDocumentSpliter
# from uniqa.components.joiners import DocumentJoiner
from uniqa.components.converters import PyPDFToDocument, JSONConverter
from uniqa.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from uniqa.components.writers import DocumentWriter
from uniqa.utils import ComponentDevice, VecsWhitening

from uniqa.document_stores.in_memory import InMemoryDocumentStore
from uniqa.components.retrievers.in_memory import InMemoryEmbeddingRetriever, InMemoryBM25Retriever

from uniqa.document_stores.milvus import MilvusDocumentStore
from uniqa.components.retrievers.milvus import MilvusEmbeddingRetriever, MilvusSparseEmbeddingRetriever
from uniqa.components.retrievers.milvus import MilvusHybridRetriever


# 白化方法(需保证数据量足够大)
is_whitening = False
if is_whitening:
    whitening_model = VecsWhitening(n_components=128)

# 初始化组件
cleaner = DocumentCleaner(
    remove_empty_lines = True,
    remove_extra_whitespaces = False,
    remove_repeated_substrings = False,
    keep_id = False,
    # remove_regex=r"\s\s+", 
    # remove_substrings = ["substring to remove"]
)

doc_embedder = SentenceTransformersDocumentEmbedder(
    model="infgrad/stella-base-zh-v3-1792d",  # dunzhang/stella-large-zh-v3-1792d
    # meta_fields_to_embed=["similar_questions"],   # 选中的元数据会拼接到文档内容中进行嵌入
    # normalize_embeddings=True,  # 向量归一化
)
text_embedder = SentenceTransformersTextEmbedder(model="infgrad/stella-base-zh-v3-1792d")
doc_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus")
writer = DocumentWriter(document_store=doc_store)

doc_embedder.warm_up()
text_embedder.warm_up()

from uniqa.components.rankers import SentenceTransformersSimilarityRanker
ranker = SentenceTransformersSimilarityRanker(
    model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", 
    # model="cross-encoder/ms-marco-MiniLM-L-6-v2", 
    # device=ComponentDevice.from_str("cpu"),  #
    # device=ComponentDevice.resolve_device(None),
)
ranker.warm_up()

from uniqa.document_stores.milvus.function import BM25BuiltInFunction
DEFAULT_CONNECTION_ARGS = {
    # "uri": "http://localhost:19530",  # 适用于 Milvus Docker 服务
    "uri": "./milvus_test.db",  # 适用于 Milvus Lite 本地
}
milvus_document_store = MilvusDocumentStore(
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

# from pymilvus.model.reranker import CrossEncoderRerankFunction, BGERerankFunction   # pip install "pymilvus[model]"
# ce_ranker = CrossEncoderRerankFunction(
#     model_name="cross-encoder/ms-marco-MiniLM-L6-v2", 
#     device="cpu" # Specify the device to use, e.g., 'cpu' or 'cuda:0'
# )
# bge_ranker = BGERerankFunction(
#     model_name="BAAI/bge-reranker-v2-m3",
#     device="cpu" # Specify the device to use, e.g., 'cpu' or 'cuda:0'
# )
from pymilvus import RRFRanker, WeightedRanker
weighted_ranker = WeightedRanker(0.5, 0.5)  # (dense, sparse)


"""
删除了long_effective字段。如果valid_begin_time、valid_end_time为空的话则认为和问题有效期相同
新增is_default_answer字段。如果一个答案没有车型标签、生效时间、最高/最低ota版本，就认为是默认答案。这是七鱼那边的一个概念，方便他们处理数据的
"""

def load_data():
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
                    {"content": "如果您已经注册了账号，您可以通过点击页面右上角的“登录”按钮，然后输入您的邮箱和密码来登录。","car_type":"理想L9/Ultra智能焕新版,理想L9/2024款 Ultra", "ota_version": "7.0.2", "valid_begin_time": None, "valid_end_time": None},
                ], 
                "category": "账户管理",
                "valid_time": datetime(2025, 7, 8).strftime('%Y-%m-%d %H:%M:%S'),  # 有效期
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
                "valid_time": datetime(2023, 12, 4).strftime('%Y-%m-%d %H:%M:%S'),  # 有效期
            },
        ),
    ]
    return docs


def build_document_store(docs):
    # 文档写入 milvus_document_store
    docs_with_embeddings = doc_embedder.run(docs)["documents"]   # meta embedding

    if is_whitening:
        # 对 Embedding 执行白化操作
        global whitening_model
        whitening_model.fit(np.array([doc.embedding for doc in docs_with_embeddings]))
        docs_with_embeddings = [
            Document(
                id=doc.id,
                content=doc.content,
                embedding=whitening_model.transform(np.array([doc.embedding])).squeeze(),
                meta=doc.meta,
            )
            for doc in docs_with_embeddings
        ]
    
    # 写入 MilvusDocumentStore
    milvus_document_store.write_documents(docs_with_embeddings)  # return int

    # # Write Documents to the DocumentStore
    # docs_with_embeddings = doc_embedder.run(docs)["documents"]
    # writer.run(docs_with_embeddings)
    # filled_document_store = writer.document_store
    # print(filled_document_store.count_documents())  # 68
    # # filled_document_store.save_to_disk("./test/documents.json")

    return milvus_document_store


def basic_hybrid_search_pipeline(query):
    # 获取知识
    docs = load_data()
    # 创建并存储文档库
    milvus_document_store = build_document_store(docs)

    # 获取问题向量
    query_embedding = text_embedder.run(query)["embedding"]
    if is_whitening:
        query_embedding = whitening_model.transform(np.array([query_embedding])).squeeze()

    # # milvus 混合检索
    # # BM25 tokenizer 可能不支持中文！！TODO
    # milvus_retriever = MilvusHybridRetriever(
    #     document_store=milvus_document_store,
    #     top_k=10, 
    #     reranker=weighted_ranker,
    #     # filters={   # Do Metadata Filtering
    #     #     "operator": "AND",
    #     #     "conditions": [
    #     #         {"field": "meta.answer[0].ota_version", "operator": ">", "value": 1.21},
    #     #         {"field": "meta.valid_time", "operator": ">", "value": datetime(2023, 11, 7)},
    #     #     ],
    #     # },
    # )
    # candidate_docs = milvus_retriever.run(
    #     query_embedding=query_embedding,
    #     query_text=query, 
    # )["documents"]

    # 语义检索
    milvus_retriever = MilvusEmbeddingRetriever(document_store=milvus_document_store, top_k=5)
    semantic_hit = milvus_retriever.run(query_embedding=query_embedding)["documents"]

    # 字面检索
    milvus_retriever = MilvusSparseEmbeddingRetriever(document_store=milvus_document_store, top_k=5)
    sparse_hit = milvus_retriever.run(query_text=query)["documents"]

    # 策略合并: 拼接+去重
    from uniqa.components.rankers import DocumentJoiner
    doc_joiner = DocumentJoiner(
        join_mode="concatenate",     # 融合方式(concatenate / merge / reciprocal_rank_fusion / distribution_based_rank_fusion)
        top_k=10,
        sort_by_score=True
    )
    candidate_docs = doc_joiner.run(documents=[semantic_hit, sparse_hit])["documents"]
    # print([x.content for x in candidate_docs])

    # 重排序
    candidate_docs_rank = ranker.run(query=query, documents=candidate_docs, top_k=10)["documents"]

    # 策略合并: 加权融合
    from uniqa.components.rankers import DocumentJoiner
    doc_joiner = DocumentJoiner(
        join_mode="merge",     # 融合方式(concatenate / merge / reciprocal_rank_fusion / distribution_based_rank_fusion)
        weights=[0.8, 0.2],    # (ranked, embedded)
        top_k=5,
        sort_by_score=True
    )
    candidate_docs = doc_joiner.run(documents=[candidate_docs_rank, semantic_hit])["documents"]
    # print([x.content for x in candidate_docs])

    # ================ InMemory ================

    # # 向量检索
    # retriever = InMemoryEmbeddingRetriever(document_store=filled_document_store, top_k=3)
    # result1 = retriever.run(query_embedding=query_embedding, scale_score=True)

    # # BM25检索可能不支持中文！！TODO
    # retriever = InMemoryBM25Retriever(document_store=filled_document_store, top_k=3)
    # result2 = retriever.run(query=query, scale_score=True)
    # # print(result2["documents"])

    # # 策略合并
    # from uniqa.components.rankers import DocumentJoiner
    # doc_joiner = DocumentJoiner(
    #     join_mode="reciprocal_rank_fusion",     # 融合方式(concatenate / merge / reciprocal_rank_fusion / distribution_based_rank_fusion)
    #     top_k=5,
    #     sort_by_score=True
    # )
    # candidate_docs = doc_joiner.run(documents=[result1["documents"], result2["documents"]])["documents"]
    # # print([x.content for x in candidate_docs])

    # # 重排序
    # candidate_docs = ranker.run(query=query, documents=candidate_docs, top_k=5)["documents"]
    # # print([x.content for x in candidate_docs])

    return candidate_docs_rank


def pretty_print_results(prediction):
    for doc in prediction:
        print(doc.id, doc.content, doc.score)
        print(doc)
        # print(doc.meta["answer"])
        print("\n", "\n")

query = "账号注册"
candidate_docs = basic_hybrid_search_pipeline(query)
pretty_print_results(candidate_docs)


