# https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/33_Hybrid_Retrieval.ipynb#scrollTo=mSUiizGNytwX

import os
import sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
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

"""在某些情况下，将有意义的元数据与文档内容一起嵌入，可能会在后续提高检索效果。"""

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
doc_embedder = SentenceTransformersDocumentEmbedder(
    model="infgrad/stella-base-zh-v3-1792d", 
    meta_fields_to_embed="date",    # 
)  # dunzhang/stella-large-zh-v3-1792d
text_embedder = SentenceTransformersTextEmbedder(model="infgrad/stella-base-zh-v3-1792d")
doc_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus")
writer = DocumentWriter(document_store=doc_store)

chunker.warm_up()
doc_embedder.warm_up()
text_embedder.warm_up()


def basic_hybrid_search_pipeline(query):

    # Preparing Documents
    docs = [
        Document(
            content="Use pip to install a basic version of Haystack's latest release: pip install farm-haystack. All the core Haystack components live in the haystack repo. But there's also the haystack-extras repo which contains components that are not as widely used, and you need to install them separately.",
            meta={"version": 1.15, "date": datetime(2023, 3, 30)},
        ),
        Document(
            content="Use pip to install a basic version of Haystack's latest release: pip install farm-haystack[inference]. All the core Haystack components live in the haystack repo. But there's also the haystack-extras repo which contains components that are not as widely used, and you need to install them separately.",
            meta={"version": 1.22, "date": datetime(2023, 11, 7)},
        ),
        Document(
            content="Use pip to install only the Haystack 2.0 code: pip install haystack-ai. The haystack-ai package is built on the main branch which is an unstable beta version, but it's useful if you want to try the new features as soon as they are merged.",
            meta={"version": 2.0, "date": datetime(2023, 12, 4)},
        ),
    ]

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

    """Comparing Retrieval With and Without Embedded Metadata"""

    query_embedding = text_embedder.run(query)["embedding"]

    # 向量检索
    retriever_with_meta = InMemoryEmbeddingRetriever(document_store=filled_document_store, top_k=3)
    candidate_docs = retriever_with_meta.run(query_embedding=query_embedding, scale_score=True)

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
    # print([x.content for x in candidate_docs])

    # # 重排序
    # ranker = SentenceTransformersSimilarityRanker(top_k=2)
    # ranker.warm_up()
    # candidate_docs = ranker.run(query=query, documents=candidate_docs)["documents"]
    # # print([x.content for x in candidate_docs])

    return candidate_docs


# query = "方向盘如何加热？"
query = "怎么联系Lynk & Co领克？"
result = basic_hybrid_search_pipeline(query)

print("Retriever with Embeddings Results:\n")
for doc in result["documents"]:
    print(doc)
