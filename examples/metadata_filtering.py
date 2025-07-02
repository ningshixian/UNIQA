# https://haystack.deepset.ai/tutorials/31_metadata_filtering

import os
import sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

from uniqa import Document
from uniqa.components.builders.answer_builder import AnswerBuilder
from uniqa.components.builders.prompt_builder import PromptBuilder
from uniqa.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from uniqa.components.generators import HuggingFaceLocalGenerator
from uniqa.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from uniqa.components.writers import DocumentWriter
from uniqa.document_stores import InMemoryDocumentStore

from uniqa.utils import ComponentDevice
from uniqa.components.converters import PyPDFToDocument, JSONConverter
# from uniqa.components.joiners import DocumentJoiner
from uniqa.components.preprocessors import DocumentCleaner
from uniqa.components.preprocessors import RecursiveDocumentSplitter, ChineseDocumentSpliter
from uniqa.components.rankers import SentenceTransformersSimilarityRanker
# from uniqa.document_stores import MilvusDocumentStore


# Preparing Documents
documents = [
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
document_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus")
document_store.write_documents(documents=documents)

# Building a Document Search Pipeline
bm25_retriever = InMemoryBM25Retriever(document_store=document_store)

# Do Metadata Filtering
query = "Haystack installation"
result = bm25_retriever.run(
    query=query, 
    filters={
        "operator": "AND",
        "conditions": [
            {"field": "meta.version", "operator": ">", "value": 1.21},
            {"field": "meta.date", "operator": ">", "value": datetime(2023, 11, 7)},
        ],
    },
    scale_score=True, 
)
print(result["documents"])
