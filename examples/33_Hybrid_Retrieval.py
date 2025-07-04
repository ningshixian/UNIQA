# https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/33_Hybrid_Retrieval.ipynb#scrollTo=mSUiizGNytwX

import os
import sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import List
from pathlib import Path

from uniqa import Document
from uniqa.components.converters import PyPDFToDocument, JSONConverter
from uniqa.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
# from uniqa.components.joiners import DocumentJoiner
from uniqa.components.preprocessors import DocumentCleaner
from uniqa.components.preprocessors import RecursiveDocumentSplitter, ChineseDocumentSpliter
from uniqa.components.retrievers.in_memory import InMemoryEmbeddingRetriever, InMemoryBM25Retriever
from uniqa.components.rankers import SentenceTransformersSimilarityRanker
# from uniqa.components.routers import FileTypeRouter
# from uniqa.components.writers import DocumentWriter
from uniqa.document_stores.in_memory import InMemoryDocumentStore
# from uniqa.document_stores import MilvusDocumentStore
from uniqa.components.writers import DocumentWriter

from uniqa.components.retrievers.milvus import MilvusEmbeddingRetriever, MilvusSparseEmbeddingRetriever
from uniqa.components.retrievers.milvus import MilvusHybridRetriever
from uniqa.document_stores.milvus import MilvusDocumentStore


# 初始化组件
cleaner = DocumentCleaner(
    remove_empty_lines = True,
    remove_extra_whitespaces = False,
    remove_repeated_substrings = False,
    keep_id = False,
    # remove_regex=r"\s\s+", 
    # remove_substrings = ["substring to remove"]
)
chunker = ChineseDocumentSpliter(
    split_by="sentence",
    split_length=5,
    split_overlap=0,
    language="zh",
    respect_sentence_boundary=False,
)
# chunker = ChineseDocumentSpliter(
#     split_by="word",
#     split_length=512,
#     split_overlap=32,
#     language="zh",
#     respect_sentence_boundary=True,   # 耗时
# )
# chunker = RecursiveDocumentSplitter(
#     separators=["\\n\\n", "sentence"]
# )
doc_embedder = SentenceTransformersDocumentEmbedder(model="infgrad/stella-base-zh-v3-1792d")  # dunzhang/stella-large-zh-v3-1792d
text_embedder = SentenceTransformersTextEmbedder(model="infgrad/stella-base-zh-v3-1792d")
doc_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus")
writer = DocumentWriter(document_store=doc_store)

chunker.warm_up()
doc_embedder.warm_up()
text_embedder.warm_up()


def basic_hybrid_search_pipeline(query):

    def load_data(doc_path: str,) -> List[Document]:
        docs = []
        samples_path = Path(doc_path)
        for path in list(samples_path.iterdir()):
            if path.is_file() and path.suffix == ".pdf":
                converter = PyPDFToDocument()
            if path.is_file() and path.suffix == ".json":
                converter = JSONConverter(
                    jq_schema=".[]", 
                    content_key="standard_question", 
                    extra_meta_fields={"id", "similar_questions", "answer", "category"}
                )
            _docs = converter.run(sources=[path])["documents"]  # [doc1, doc2, ...]
            docs.extend(_docs)
        return docs

    # 读取文档数据
    doc_path = os.path.dirname(os.path.abspath(__file__)) + "/test_documents"
    docs = load_data(doc_path)

    # 切片
    docs = cleaner.run(documents=docs)["documents"]
    doc_chunks = chunker.run(docs)["documents"]
    doc_chunks = doc_chunks[:50]    # 
    # for d in doc_chunks:
    #     print("---->" + d.content)

    # Write Documents to the DocumentStore
    docs_with_embeddings = doc_embedder.run(doc_chunks)["documents"]
    writer.run(docs_with_embeddings)
    filled_document_store = writer.document_store
    print(filled_document_store.count_documents())  # 68
    # filled_document_store.save_to_disk("./test/documents.json")

    # 

    query_embedding = text_embedder.run(query)["embedding"]

    # 向量检索
    retriever = InMemoryEmbeddingRetriever(document_store=filled_document_store, top_k=3)
    result1 = retriever.run(query_embedding=query_embedding, scale_score=True)

    # BM25检索可能不支持中文！！TODO
    retriever = InMemoryBM25Retriever(document_store=filled_document_store, top_k=3)
    result2 = retriever.run(query=query, scale_score=True)
    # print(result2["documents"])

    # 策略合并
    from uniqa.components.rankers import DocumentJoiner
    doc_joiner = DocumentJoiner(
        join_mode="reciprocal_rank_fusion",     # 融合方式(concatenate / merge / reciprocal_rank_fusion / distribution_based_rank_fusion)
        top_k=5,
        sort_by_score=True
    )
    candidate_docs = doc_joiner.run(documents=[result1["documents"], result2["documents"]])["documents"]
    print([x.content for x in candidate_docs])

    # 重排序
    ranker = SentenceTransformersSimilarityRanker(top_k=2)
    ranker.warm_up()
    candidate_docs = ranker.run(query=query, documents=candidate_docs)["documents"]
    print([x.content for x in candidate_docs])

    return candidate_docs


# def pretty_print_results(prediction):
#     for doc in prediction:
#         print(doc.meta["title"], "\t", doc.score)
#         print(doc.meta["abstract"])
#         print("\n", "\n")

# query = "方向盘如何加热？"
query = "怎么联系Lynk & Co领克？"
candidate_docs = basic_hybrid_search_pipeline(query)
# pretty_print_results(candidate_docs)

"""
['驾乘人员造成人身伤害或导致车辆损坏。\n联系Lynk&Co领克\n如果您对本手册的内容有疑问，请通过以下方式联系Lynk&Co领克。\nLynk&Co领克客户联络中心\n客户服务热线：4006-010101\nLynk&Co领克\nhttp://www.lynkco.com\n事件数据记录系统\nLynk&Co领克汽车配备有事件数据记录系统（EDR），该系统用于记\n录车辆发生碰撞事故的相关信息，如车辆的行驶速度（表示事件发生\n时刻车辆速度），制动状态（表示事件发生时刻车辆是否制动）。\n', '因此产生的任何问题，Lynk&Co领克将不会\n承担责任。\n您可以订购某些经过Lynk&Co领克认证的选装装备。\n由于销售市场不\n同，专门为某些国家或地区提供的选装装备可能不适用于您的地区。\n在您订购选装装备之前，请查询当地法律法规，并联系Lynk&Co领克\n中心。\n如果您使用了未经Lynk&Co领克认证或不适用您所在地区的选\n装装备，引起的任何问题，Lynk&Co领克将不会承担责任。\n']
"""
