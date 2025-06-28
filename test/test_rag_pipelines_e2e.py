# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import json
from pathlib import Path

from uniqa import Document
from uniqa.components.builders.answer_builder import AnswerBuilder
from uniqa.components.builders.prompt_builder import PromptBuilder
from uniqa.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from uniqa.components.generators import HuggingFaceLocalGenerator
from uniqa.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from uniqa.components.writers import DocumentWriter
from uniqa.document_stores import InMemoryDocumentStore

from uniqa.components.converters import PyPDFToDocument, TextFileToDocument
# from uniqa.components.joiners import DocumentJoiner
from uniqa.components.preprocessors import DocumentCleaner, RecursiveDocumentSplitter
from uniqa.components.rankers import SentenceTransformersSimilarityRanker
# from uniqa.document_stores import MilvusDocumentStore

"""
pytest -svx test/test_rag_pipelines_e2e.py
pytest -svx test/test_rag_pipelines_e2e.py::test_rag_pipelines_e2e
"""


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

# 读取文件
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


def test_rag_pipeline():
    # Create the RAG pipeline
    prompt_template = """
    Given these documents, answer the question.\nDocuments:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{question}}
    \nAnswer:
    """

    doc_chunks = chunker.run(docs)["documents"]
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

    # 混合检索
    candidates = result1["documents"] + result2["documents"]
    ranker = SentenceTransformersSimilarityRanker(top_k=2)
    ranker.warm_up()
    result = ranker.run(query=query, documents=candidates)
    candidate_docs = result["documents"]
    print(candidate_docs[0].content)

    # 提示模板
    prompt_builder = PromptBuilder(template=prompt_template)
    prompt = prompt_builder.run(documents=candidate_docs, question=query)["prompt"]

    # 生成器
    llm = HuggingFaceLocalGenerator(
        model="Qwen/Qwen3-0.6B",
        task="text-generation",
        generation_kwargs={"max_new_tokens": 500, "temperature": 0.9}
    )
    llm.warm_up()
    replies = llm.run(prompt)["replies"]

    # 解析答案
    answer_builder = AnswerBuilder(pattern="Answer: (.*)")
    answers = answer_builder.run(
        query=query, 
        replies=replies
    )["answers"]
    print(answers)


test_rag_pipeline()


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_bm25_rag_pipeline(tmp_path):
    # Create the RAG pipeline
    prompt_template = """
    Given these documents, answer the question.\nDocuments:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{question}}
    \nAnswer:
    """
    rag_pipeline = Pipeline()
    rag_pipeline.add_component(instance=InMemoryBM25Retriever(document_store=InMemoryDocumentStore()), name="retriever")
    rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
    rag_pipeline.add_component(instance=OpenAIGenerator(), name="llm")
    rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("llm.meta", "answer_builder.meta")
    rag_pipeline.connect("retriever", "answer_builder.documents")

    # Serialize the pipeline to YAML
    with open(tmp_path / "test_bm25_rag_pipeline.yaml", "w") as f:
        rag_pipeline.dump(f)

    # Load the pipeline back
    with open(tmp_path / "test_bm25_rag_pipeline.yaml", "r") as f:
        rag_pipeline = Pipeline.load(f)

    # Populate the document store
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]
    rag_pipeline.get_component("retriever").document_store.write_documents(documents)

    # Query and assert
    questions = ["Who lives in Paris?", "Who lives in Berlin?", "Who lives in Rome?"]
    answers_spywords = ["Jean", "Mark", "Giorgio"]

    for question, spyword in zip(questions, answers_spywords):
        result = rag_pipeline.run(
            {
                "retriever": {"query": question},
                "prompt_builder": {"question": question},
                "answer_builder": {"query": question},
            }
        )

        assert len(result["answer_builder"]["answers"]) == 1
        generated_answer = result["answer_builder"]["answers"][0]
        assert spyword in generated_answer.data
        assert generated_answer.query == question
        assert hasattr(generated_answer, "documents")
        assert hasattr(generated_answer, "meta")


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_embedding_retrieval_rag_pipeline(tmp_path):
    # Create the RAG pipeline
    prompt_template = """
    Given these documents, answer the question.\nDocuments:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{question}}
    \nAnswer:
    """
    rag_pipeline = Pipeline()
    rag_pipeline.add_component(
        instance=SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"), name="text_embedder"
    )
    rag_pipeline.add_component(
        instance=InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore()), name="retriever"
    )
    rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
    rag_pipeline.add_component(instance=OpenAIGenerator(), name="llm")
    rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
    rag_pipeline.connect("text_embedder", "retriever")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("llm.meta", "answer_builder.meta")
    rag_pipeline.connect("retriever", "answer_builder.documents")

    # Serialize the pipeline to JSON
    with open(tmp_path / "test_embedding_rag_pipeline.json", "w") as f:
        json.dump(rag_pipeline.to_dict(), f)

    # Load the pipeline back
    with open(tmp_path / "test_embedding_rag_pipeline.json", "r") as f:
        rag_pipeline = Pipeline.from_dict(json.load(f))

    # Populate the document store
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]
    document_store = rag_pipeline.get_component("retriever").document_store
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(
        instance=SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
        name="document_embedder",
    )
    indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="document_writer")
    indexing_pipeline.connect("document_embedder", "document_writer")
    indexing_pipeline.run({"document_embedder": {"documents": documents}})

    # Query and assert
    questions = ["Who lives in Paris?", "Who lives in Berlin?", "Who lives in Rome?"]
    answers_spywords = ["Jean", "Mark", "Giorgio"]

    for question, spyword in zip(questions, answers_spywords):
        result = rag_pipeline.run(
            {
                "text_embedder": {"text": question},
                "prompt_builder": {"question": question},
                "answer_builder": {"query": question},
            }
        )

        assert len(result["answer_builder"]["answers"]) == 1
        generated_answer = result["answer_builder"]["answers"][0]
        assert spyword in generated_answer.data
        assert generated_answer.query == question
        assert hasattr(generated_answer, "documents")
        assert hasattr(generated_answer, "meta")
