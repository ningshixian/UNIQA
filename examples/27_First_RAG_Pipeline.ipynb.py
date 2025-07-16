# 基础 RAG 问答功能
# https://haystack.deepset.ai/tutorials/27_first_rag_pipeline
# https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/27_First_RAG_Pipeline.ipynb

import os
import sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import List
from pathlib import Path
import traceback

from uniqa import Document
from uniqa.components.builders.answer_builder import AnswerBuilder
from uniqa.components.builders.prompt_builder import PromptBuilder
from uniqa.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from uniqa.components.generators import HuggingFaceLocalGenerator, OpenAIGenerator
# from uniqa.components.generators.chat import HuggingFaceLocalChatGenerator, OpenAIChatGenerator
from uniqa.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from uniqa.components.writers import DocumentWriter
from uniqa.document_stores.in_memory import InMemoryDocumentStore

from uniqa.utils import ComponentDevice
from uniqa.components.converters import PyPDFToDocument, JSONConverter
# from uniqa.components.joiners import DocumentJoiner
from uniqa.components.preprocessors import DocumentCleaner
from uniqa.components.preprocessors import RecursiveDocumentSplitter, ChineseDocumentSpliter
from uniqa.components.rankers import SentenceTransformersSimilarityRanker
# from uniqa.document_stores import MilvusDocumentStore

from haystack.dataclasses import StreamingChunk
from haystack.dataclasses import ComponentInfo, StreamingCallbackT, AsyncStreamingCallbackT
from haystack.dataclasses import select_streaming_callback
from haystack.components.generators.utils import print_streaming_chunk


"""
pytest -svx test/pipeline_rag.py
"""

# 初始化组件
document_cleaner = DocumentCleaner(
    remove_empty_lines = True,
    remove_extra_whitespaces = False,
    remove_repeated_substrings = False,
    keep_id = False,
    # remove_regex=r"\s\s+", 
    # remove_substrings = ["substring to remove"]
)
document_splitter = ChineseDocumentSpliter(
    split_by="sentence",
    split_length=5,
    split_overlap=0,
    language="zh",
    respect_sentence_boundary=False,
)
# document_splitter = ChineseDocumentSpliter(
#     split_by="word",
#     split_length=512,
#     split_overlap=32,
#     language="zh",
#     respect_sentence_boundary=True,   # 耗时
# )
# document_splitter = RecursiveDocumentSplitter(
#     separators=["\\n\\n", "sentence"]
# )
doc_embedder = SentenceTransformersDocumentEmbedder(model="infgrad/stella-base-zh-v3-1792d")  # dunzhang/stella-large-zh-v3-1792d
text_embedder = SentenceTransformersTextEmbedder(model="infgrad/stella-base-zh-v3-1792d")
doc_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus")
writer = DocumentWriter(document_store=doc_store)

# # 定义一个异步流式回调函数
# async def async_print_streaming_chunk(chunk: StreamingChunk):
#     print_streaming_chunk(chunk)

# # 创建OpenAI ChatGenerator (支持异步生成)
# openai_api_key = os.environ["OPENAI_API_KEY"]
# openai_api_base = os.environ["OPENAI_API_BASE"]
# chat_llm = OpenAIGenerator(
#     model='qwen__qwen3-235b-a22b', 
#     api_base_url=openai_api_base, 
#     generation_kwargs={
#         "temperature": 0.6,
#         "top_p": 0.7,
#         "max_tokens": 512,   # The maximum number of tokens the output text can have.
#         "frequency_penalty": 1.2,
#     }, 
#     timeout=60, 
#     max_retries=2,
#     streaming_callback=print_streaming_chunk,    # 流式回调
#     # streaming_callback=async_print_streaming_chunk,    # 异步回调 → stream=True
# )

# 创建Generator (支持流式输出)
llm = HuggingFaceLocalGenerator(
    model="Qwen/Qwen3-0.6B",
    task="text-generation",
    device=ComponentDevice.from_str("cpu"),  #
    # device=ComponentDevice.resolve_device(None),
    generation_kwargs={   # transformers generation()参数
        "max_new_tokens": 512,   # input prompt + max_new_tokens
        "temperature": 0.9,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
    },
    streaming_callback=print_streaming_chunk,    # 流式回调处理函数
)
llm.warm_up()

document_splitter.warm_up()
doc_embedder.warm_up()
text_embedder.warm_up()

from uniqa.components.rankers import SentenceTransformersSimilarityRanker
ranker = SentenceTransformersSimilarityRanker(
    model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", 
    # model="cross-encoder/ms-marco-MiniLM-L-6-v2", 
    # device=ComponentDevice.from_str("cpu"),  #
    # device=ComponentDevice.resolve_device(None),
    scale_score=True,   # 得分归一化 - sigmoid
)
ranker.warm_up()


def basic_rag_pipeline(query):

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

    # 加载数据集并转换为文档
    doc_path = os.path.dirname(os.path.abspath(__file__)) + "/test_documents"
    docs = load_data(doc_path)

    # 文档切片
    docs = document_cleaner.run(documents=docs)["documents"]
    doc_chunks = document_splitter.run(documents=docs)["documents"]
    doc_chunks = doc_chunks[:50]    # 
    # for d in doc_chunks:
    #     print("---->" + d.content)

    # 嵌入文档
    docs_with_embeddings = doc_embedder.run(doc_chunks)["documents"]
    writer.run(docs_with_embeddings)    # 同 doc_store.write_documents(docs_with_embeddings)
    filled_document_store = writer.document_store
    print(filled_document_store.count_documents())  # 68
    # filled_document_store.save_to_disk("./test/documents.json")

    # print("check point 1")

    # 构建检索问答 / 构建 RAG 流程
    query_embedding = text_embedder.run(query)["embedding"]

    # 两路检索器召回
    vector_retriever = InMemoryEmbeddingRetriever(document_store=filled_document_store, top_k=5)
    result1 = vector_retriever.run(query_embedding=query_embedding, scale_score=True)
    bm25_retriever = InMemoryBM25Retriever(document_store=filled_document_store, top_k=5)   # 导入 hanlp 用于支持中文分词
    result2 = bm25_retriever.run(query=query, scale_score=True)

    # 策略合并 - joiner
    from uniqa.components.rankers import DocumentJoiner
    doc_joiner = DocumentJoiner(
        join_mode="reciprocal_rank_fusion",     # 融合方式(concatenate / merge / reciprocal_rank_fusion / distribution_based_rank_fusion)
        top_k=5,
        sort_by_score=True
    )
    candidate_docs = doc_joiner.run(documents=[result1["documents"], result2["documents"]])["documents"]

    # 重排序
    candidate_docs = ranker.run(query=query, documents=candidate_docs, top_k=3)["documents"]
    # print([x.content for x in candidate_docs])

    # 创建提示模板（使用了标准的jinja格式的prompt模板）
    prompt_template = """
    根据以下信息，使用中文回答问题。

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """
    prompt_builder = PromptBuilder(template=prompt_template)
    prompt = prompt_builder.run(documents=candidate_docs, question=query)["prompt"]
    
    # print("check point 2")

    # ....检索逻辑....

    try:

        # 流式生成回复
        replies = llm.run(prompt)["replies"]
        # print(replies[0])

        # # 解析答案
        # answer_builder = AnswerBuilder(pattern="Answer: (.*)")
        # answers = answer_builder.run(
        #     query=query, 
        #     replies=replies
        # )["answers"]
        return replies[0]
    
    except Exception as e:
        traceback.print_exc()
        return f"data: 错误：{str(e)}\n\n"


# Asking a Question
query = "怎么联系Lynk & Co领克？"
answers = basic_rag_pipeline(query)
print("\n", answers)
