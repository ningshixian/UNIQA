# Building a agentic RAG with Function Calling
# https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/40_Building_Chat_Application_with_Function_Calling.ipynb#scrollTo=ZE0SEGY92GHJ

import os
import sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import List
from pathlib import Path

from uniqa import Document
from uniqa.components.builders.answer_builder import AnswerBuilder
from uniqa.components.builders.chat_prompt_builder import ChatPromptBuilder
from uniqa.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from uniqa.components.generators.chat import HuggingFaceLocalChatGenerator
from uniqa.components.writers import DocumentWriter
from uniqa.dataclasses import ChatMessage, ToolCall

from uniqa.utils import ComponentDevice
from uniqa.components.converters import PyPDFToDocument, JSONConverter
# from uniqa.components.joiners import DocumentJoiner
from uniqa.components.preprocessors import DocumentCleaner
from uniqa.components.preprocessors import RecursiveDocumentSplitter, ChineseDocumentSpliter
from uniqa.components.rankers import SentenceTransformersSimilarityRanker

from uniqa.document_stores.in_memory import InMemoryDocumentStore
from uniqa.components.retrievers.in_memory import InMemoryEmbeddingRetriever, InMemoryBM25Retriever

from uniqa.document_stores.milvus import MilvusDocumentStore
from uniqa.components.retrievers.milvus import MilvusEmbeddingRetriever, MilvusSparseEmbeddingRetriever
from uniqa.components.retrievers.milvus import MilvusHybridRetriever


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

doc_embedder = SentenceTransformersDocumentEmbedder(
    model="infgrad/stella-base-zh-v3-1792d",  # dunzhang/stella-large-zh-v3-1792d
    # meta_fields_to_embed=["similar_questions"],   # 选中的元数据会拼接到文档内容中进行嵌入
    # normalize_embeddings=True,  # 向量归一化
)
text_embedder = SentenceTransformersTextEmbedder(model="infgrad/stella-base-zh-v3-1792d")
doc_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus")
writer = DocumentWriter(document_store=doc_store)

document_splitter.warm_up()
doc_embedder.warm_up()
text_embedder.warm_up()

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

from uniqa.components.rankers import SentenceTransformersSimilarityRanker
ranker = SentenceTransformersSimilarityRanker(
    model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", 
    # model="cross-encoder/ms-marco-MiniLM-L-6-v2", 
    # device=ComponentDevice.from_str("cpu"),  #
    # device=ComponentDevice.resolve_device(None),
    top_k=5, 
)
ranker.warm_up()

def basic_rag_pipeline(query):

    docs = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
        Document(content="My name is Marta and I live in Madrid."),
        Document(content="My name is Harry and I live in London."),
    ]

    query_embedding = text_embedder.run(query)["embedding"]

    """Create the querying pipelines"""

    # 文档写入 milvus_document_store
    docs_with_embeddings = doc_embedder.run(docs)["documents"]
    milvus_document_store.write_documents(docs_with_embeddings)  # return int

    # # milvus 混合检索
    # # BM25 tokenizer 可能不支持中文！！TODO
    # milvus_retriever = MilvusHybridRetriever(
    #     document_store=milvus_document_store,
    #     top_k=10, 
    #     reranker=None,
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

    # 

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

    # Define a Template Prompt
    template = [
        ChatMessage.from_system(
            """
    Answer the questions based on the given context.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}
    Question: {{ question }}
    Answer:
    """
        )
    ]
    prompt_builder = ChatPromptBuilder(template=template)
    prompt = prompt_builder.run(documents=candidate_docs, question=query)["prompt"]
    # print("===ChatMessage=== ", prompt)

    # 初始化生成器
    llm = HuggingFaceLocalChatGenerator(
        model="Qwen/Qwen3-0.6B",
        task="text-generation",
        device=ComponentDevice.from_str("cpu"),  #
        # device=ComponentDevice.resolve_device(None),
        generation_kwargs={
            "max_new_tokens": 32768,   # input prompt + max_new_tokens
            "temperature": 0.9,
            "top_p": 0.95,
        }
    )
    llm.warm_up()
    replies = llm.run(
        prompt, 
        generation_kwargs = {
            "repetition_penalty": 1.2,
        }   # transformers generation()参数
    )["replies"]
    # print("===Generator Output=== ", replies[0])

    # 解析答案
    pattern = r'(?:<\/think>\n+|Answer:\s*)(.*)'
    answer_builder = AnswerBuilder(pattern=pattern)
    answers = answer_builder.run(
        query=query, 
        replies=replies
    )["answers"]
    return answers


# # Test: Asking a Question
# query = {'query': "Who lives in London?"}
# answers = basic_rag_pipeline(**query)
# print(answers)
# exit()


"""Building a agentic RAG with Function Calling"""


from typing import Annotated, Literal
from haystack.tools import create_tool_from_function    #
from uniqa.tools import Tool
from uniqa.tools.tool_invoker import ToolInvoker


# Creating a Tool from Function
parameters = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The query to use in the search. Infer this from the user's message. It should be a question or a statement",
        }
    },
    "required": ["query"],
}

rag_pipeline_tool = Tool(
    name="rag_pipeline_tool",
    description="Get information about where people live",
    parameters=parameters,
    function=basic_rag_pipeline,
)

WEATHER_INFO = {
    "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
    "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
    "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
    "Madrid": {"weather": "sunny", "temperature": 10, "unit": "celsius"},
    "London": {"weather": "cloudy", "temperature": 9, "unit": "celsius"},
}


def get_weather(
    city: Annotated[str, "the city for which to get the weather"] = "Berlin",
    unit: Annotated[Literal["Celsius", "Fahrenheit"], "the unit for the temperature"] = "Celsius",
):
    """A simple function to get the current weather for a location."""
    if city in WEATHER_INFO:
        return WEATHER_INFO[city]
    else:
        return {"weather": "sunny", "temperature": 21.8, "unit": "fahrenheit"}


# 函数封装成工具类 Tool
weather_tool = create_tool_from_function(get_weather)


import gradio as gr
import json

# 执行工具，并将结果拼接回推理链中，直到 finish_reason=stop
tool_invoker = ToolInvoker(tools=[rag_pipeline_tool, weather_tool])

# # 测试 tool_invoker
# test_replies = [ChatMessage(_role='assistant', _content=[ToolCall(tool_name='rag_pipeline_tool', arguments={'query': "What's the weather like where Mark lives?"}, id=None)], _name=None, _meta={'finish_reason': 'tool_calls', 'index': 0, 'model': 'Qwen/Qwen3-0.6B', 'usage': {'completion_tokens': 138, 'prompt_tokens': 304, 'total_tokens': 442}})]
# answers = tool_invoker.run(messages=test_replies)["tool_messages"]
# print(answers)
# exit()


chat_generator = HuggingFaceLocalChatGenerator(
    model="Qwen/Qwen3-0.6B",
    task="text-generation",
    device=ComponentDevice.from_str("cpu"),  #
    # device=ComponentDevice.resolve_device(None),
    generation_kwargs={
        "max_new_tokens": 32768,   # input prompt + max_new_tokens
        "temperature": 0.9,
        "top_p": 0.95,
    },
    tools=[rag_pipeline_tool, weather_tool]  # Running HuggingFaceLocalChatGenerator with Tools
)
chat_generator.warm_up()

response = None
messages = [
    ChatMessage.from_system(
        "Use the tools that you're provided with. Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
    ),
    # ChatMessage.from_user("Can you tell me where Mark lives? What's the weather like in Berlin?"),
]


def chatbot_with_tc(message, history):
    global messages
    messages.append(ChatMessage.from_user(message))
    response = chat_generator.run(messages=messages)
    # print(response)

    while True:
        print("==========", response["replies"])

        # if OpenAI response is a function call
        if response and response["replies"][0].tool_calls:
            tool_result_messages = tool_invoker.run(messages=response["replies"])["tool_messages"]
            # Pass all the messages to the ChatGenerator with the correct order
            messages = messages + response["replies"] + tool_result_messages
            response = chat_generator.run(messages=messages)

        # Regular Conversation
        else:
            # final_replies = response["replies"]
            # print(f"~~~ final replies: {final_replies}")
            messages.append(response["replies"][0])
            break
    return response["replies"][0].text


# Test: Asking a Question
query = {'message': "Tell me where Giorgio lives? I want to know the weather there."}
answers = chatbot_with_tc(**query, history=[])
print(answers)
exit()


demo = gr.ChatInterface(
    fn=chatbot_with_tc,
    type="messages",
    examples=[
        "Can you tell me where Giorgio lives?",
        "What's the weather like in Madrid?",
        "Who lives in London?",
        "What's the weather like where Mark lives?",
    ],
    title="Ask me about weather or where people live!",
    theme=gr.themes.Ocean(),
)

# Uncomment the line below to launch the chat app with UI
demo.launch()

