# create an agentic RAG pipeline with conditional routing that can fallback to websearch if the answer is not present in your dataset.
# https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/36_Building_Fallbacks_with_Conditional_Routing.ipynb

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
from uniqa.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from uniqa.components.writers import DocumentWriter
from uniqa.document_stores.in_memory import InMemoryDocumentStore
from uniqa.dataclasses import ChatMessage

from uniqa.utils import ComponentDevice
from uniqa.components.converters import PyPDFToDocument, JSONConverter
# from uniqa.components.joiners import DocumentJoiner
from uniqa.components.preprocessors import DocumentCleaner
from uniqa.components.preprocessors import RecursiveDocumentSplitter, ChineseDocumentSpliter
from uniqa.components.rankers import SentenceTransformersSimilarityRanker
# from uniqa.document_stores import MilvusDocumentStore

from uniqa.tools import Tool
from uniqa.tools.tool_invoker import ToolInvoker


doc_embedder = SentenceTransformersDocumentEmbedder(model="infgrad/stella-base-zh-v3-1792d")  # dunzhang/stella-large-zh-v3-1792d
text_embedder = SentenceTransformersTextEmbedder(model="infgrad/stella-base-zh-v3-1792d")
document_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus")
# writer = DocumentWriter(document_store=document_store)

doc_embedder.warm_up()
text_embedder.warm_up()


documents = [
    Document(
        content="""Munich, the vibrant capital of Bavaria in southern Germany, exudes a perfect blend of rich cultural
                                heritage and modern urban sophistication. Nestled along the banks of the Isar River, Munich is renowned
                                for its splendid architecture, including the iconic Neues Rathaus (New Town Hall) at Marienplatz and
                                the grandeur of Nymphenburg Palace. The city is a haven for art enthusiasts, with world-class museums like the
                                Alte Pinakothek housing masterpieces by renowned artists. Munich is also famous for its lively beer gardens, where
                                locals and tourists gather to enjoy the city's famed beers and traditional Bavarian cuisine. The city's annual
                                Oktoberfest celebration, the world's largest beer festival, attracts millions of visitors from around the globe.
                                Beyond its cultural and culinary delights, Munich offers picturesque parks like the English Garden, providing a
                                serene escape within the heart of the bustling metropolis. Visitors are charmed by Munich's warm hospitality,
                                making it a must-visit destination for travelers seeking a taste of both old-world charm and contemporary allure."""
    )
]

document_store.write_documents(documents)

"""## Creating the Initial RAG Pipeline Components"""

# query = "Where is Munich?"
query = "How many people live in Munich?"

retriever = InMemoryBM25Retriever(document_store)
candidate_docs = retriever.run(query)["documents"]

prompt_template = [
    ChatMessage.from_user(
        """
Answer the following query given the documents.
If the answer is not contained within the documents reply with 'no_answer'

Documents:
{% for document in documents %}
  {{document.content}}
{% endfor %}
Query: {{query}}
"""
    )
]

prompt_builder = ChatPromptBuilder(template=prompt_template, required_variables="*")
prompt = prompt_builder.run(documents=candidate_docs, query=query)["prompt"]
print(prompt)

# 初始化生成器
llm = HuggingFaceLocalChatGenerator(
    model="Qwen/Qwen3-0.6B",
    task="text-generation",
    device=ComponentDevice.from_str("cpu"),  #
    # device=ComponentDevice.resolve_device(None),
    generation_kwargs={
        "max_new_tokens": 512,   # input prompt + max_new_tokens
        "temperature": 0.9,
        "top_p": 0.95,
    }
)
llm.warm_up()

replies = llm.run(
    prompt, 
    generation_kwargs = {"repetition_penalty": 1.2,}   # transformers generation()参数
)["replies"]
print(replies[0])

"""## Initializing the Web-RAG Components"""

from haystack.components.websearch.searchapi import SearchApiWebSearch

prompt_for_websearch = [
    ChatMessage.from_user(
        """
Answer the following query given the documents retrieved from the web.
Your answer should indicate that your answer was generated from websearch.

Documents:
{% for document in documents %}
  {{document.content}}
{% endfor %}

Query: {{query}}
"""
    )
]

# 设置环境变量
import os
os.environ["SEARCHAPI_API_KEY"] = "Zie7hh93PcACAdBp2sTQSzgP"

websearch = SearchApiWebSearch()
prompt_builder_for_websearch = ChatPromptBuilder(template=prompt_for_websearch, required_variables="*")
llm_for_websearch = HuggingFaceLocalChatGenerator(
    model="Qwen/Qwen3-0.6B",
    task="text-generation",
    device=ComponentDevice.from_str("cpu"),  #
    # device=ComponentDevice.resolve_device(None),
    generation_kwargs={
        "max_new_tokens": 512,   # input prompt + max_new_tokens
        "temperature": 0.9,
        "top_p": 0.95,
    }
)
llm_for_websearch.warm_up()

# from haystack.components.routers import ConditionalRouter

if 'no_answer' in replies[0].text:
    print("No answer found. Trying to find an answer in the web...")
    search_results = websearch.run(query)["documents"]
    prompt2 = prompt_builder_for_websearch.run(documents=search_results, query=query)["prompt"]
    replies2 = llm_for_websearch.run(prompt2)["replies"]
    print(replies2[0].text)
else:
    print(replies[0].text)


# query = "Where is Munich?"
# """✅ The answer to this query can be found in the defined document.

# query = "How many people live in Munich?"
# """If you check the whole result, you will see that `websearch` component also provides links to Documents retrieved from the web:"""
