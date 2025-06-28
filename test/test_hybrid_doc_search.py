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
from uniqa.components.converters import PyPDFToDocument, TextFileToDocument
from uniqa.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
# from uniqa.components.joiners import DocumentJoiner
from uniqa.components.preprocessors import DocumentCleaner, RecursiveDocumentSplitter
from uniqa.components.retrievers.in_memory import InMemoryEmbeddingRetriever, InMemoryBM25Retriever
from uniqa.components.rankers import SentenceTransformersSimilarityRanker
# from uniqa.components.routers import FileTypeRouter
# from uniqa.components.writers import DocumentWriter
from uniqa.document_stores import InMemoryDocumentStore
# from uniqa.document_stores import MilvusDocumentStore
from uniqa.components.writers import DocumentWriter


cleaner = DocumentCleaner(remove_substrings = ["substring to remove"])
chunker = RecursiveDocumentSplitter(split_length=260, split_overlap=0, separators=["\\n\\n", "\\n", ".", " "])
chunker.warm_up()
doc_embedder = SentenceTransformersDocumentEmbedder(model="infgrad/stella-base-zh-v3-1792d")  # dunzhang/stella-large-zh-v3-1792d
text_embedder = SentenceTransformersTextEmbedder(model="infgrad/stella-base-zh-v3-1792d")
doc_embedder.warm_up()
text_embedder.warm_up()
doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
# milvus_doc_store = MilvusDocumentStore()
writer = DocumentWriter(document_store=doc_store)


docs = []
samples_path = Path(os.path.dirname(os.path.abspath(__file__)) + "/test_documents")
for path in list(samples_path.iterdir()):
    if path.is_file() and path.suffix == ".pdf":
        converter = PyPDFToDocument()
    if path.is_file() and path.suffix == ".txt":
        converter = TextFileToDocument()
    
    _docs = converter.run(sources=[path])["documents"]  # [doc1, doc2, ...]
    _docs = cleaner.run(documents=_docs)["documents"]
    # print(_docs[0].content[:512])
    docs.extend(_docs)


doc_chunks = chunker.run(docs)["documents"]
# print(doc_chunks[:5])

result = doc_embedder.run(doc_chunks)["documents"]
writer.run(result)

# Create the querying pipelines
query = "方向盘如何加热？"
query_embedding = text_embedder.run(query)["embedding"]

filled_document_store = writer.document_store
print(filled_document_store.count_documents())  # 68
# filled_document_store.save_to_disk("./test/documents.json")
retriever = InMemoryEmbeddingRetriever(document_store=filled_document_store, top_k=3)
result1 = retriever.run(query_embedding=query_embedding, scale_score=True)
print(result1["documents"])

# BM25检索可能不支持中文！！TODO
retriever = InMemoryBM25Retriever(document_store=filled_document_store, top_k=3)
result2 = retriever.run(query=query)
print(result2["documents"])

candidates = result1["documents"] + result2["documents"]
ranker = SentenceTransformersSimilarityRanker(top_k=2)
ranker.warm_up()
result = ranker.run(query=query, documents=candidates)
docs = result["documents"]
print(docs[0].content)


