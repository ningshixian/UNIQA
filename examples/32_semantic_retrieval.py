# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

# from uniqa import Pipeline
from uniqa.components.converters import PyPDFToDocument, JSONConverter
from uniqa.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
# from uniqa.components.joiners import DocumentJoiner
from uniqa.components.preprocessors import DocumentCleaner
from uniqa.components.preprocessors import RecursiveDocumentSplitter, ChineseDocumentSpliter
from uniqa.components.retrievers.in_memory import InMemoryEmbeddingRetriever, InMemoryBM25Retriever
from uniqa.components.rankers import SentenceTransformersSimilarityRanker
# from uniqa.components.routers import FileTypeRouter
# from uniqa.components.writers import DocumentWriter
from uniqa.document_stores import InMemoryDocumentStore
# from uniqa.document_stores import MilvusDocumentStore
from uniqa.components.writers import DocumentWriter


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
    split_length=10,
    split_overlap=0,
    language="zh",
    respect_sentence_boundary=False,
)
# chunker = ChineseDocumentSpliter(
#     split_by="word",
#     split_length=200,
#     split_overlap=0,
#     language="zh",
#     respect_sentence_boundary=True,
# )
# chunker = RecursiveDocumentSplitter(
#     separators=["\\n\\n", "sentence"]
# )
chunker.warm_up()

doc_embedder = SentenceTransformersDocumentEmbedder(model="infgrad/stella-base-zh-v3-1792d")  # dunzhang/stella-large-zh-v3-1792d
text_embedder = SentenceTransformersTextEmbedder(model="infgrad/stella-base-zh-v3-1792d")
doc_embedder.warm_up()
text_embedder.warm_up()

doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
writer = DocumentWriter(document_store=doc_store)


def basic_hybrid_search_pipeline(query):

    docs = []
    samples_path = Path(os.path.dirname(os.path.abspath(__file__)) + "/test_documents")
    for path in list(samples_path.iterdir()):
        if path.is_file() and path.suffix == ".pdf":
            converter = PyPDFToDocument()
        if path.is_file() and path.suffix == ".json":
            converter = JSONConverter(
                jq_schema=".[]", 
                content_key="standard_question", 
                extra_meta_fields={"id", "similar_questions", "answer", "category"}
            )
        # 读取文件
        _docs = converter.run(sources=[path])["documents"]  # [doc1, doc2, ...]
        docs.extend(_docs)


    _docs = cleaner.run(documents=_docs)["documents"]
    docs.extend(_docs)
    doc_chunks = chunker.run(docs)["documents"]
    doc_chunks = doc_chunks[:50]
    # for d in doc_chunks:
    #     print("---->" + d.content)

    # 创建向量
    docs_with_embeddings = doc_embedder.run(doc_chunks)["documents"]
    writer.run(docs_with_embeddings)
    query_embedding = text_embedder.run(query)["embedding"]

    # 向量检索
    filled_document_store = writer.document_store
    print(filled_document_store.count_documents())  # 68
    # filled_document_store.save_to_disk("./test/documents.json")
    retriever = InMemoryEmbeddingRetriever(document_store=filled_document_store, top_k=3)
    candidate_docs = retriever.run(query_embedding=query_embedding, scale_score=True)

    return candidate_docs


# query = "方向盘如何加热？"
query = "怎么联系Lynk & Co领克？"
candidate_docs = basic_hybrid_search_pipeline()(query)
print(candidate_docs)
