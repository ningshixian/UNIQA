# retrievers 模块的任务是 检索文档。

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    # "auto_merging_retriever": ["AutoMergingRetriever"],
    # "filter_retriever": ["FilterRetriever"],
    "in_memory": ["InMemoryBM25Retriever", "InMemoryEmbeddingRetriever"],
    # "sentence_window_retriever": ["SentenceWindowRetriever"],
    # "indexs": ["FaissIndex", "MilvusIndex"],
}

# from .auto_merging_retriever import AutoMergingRetriever as AutoMergingRetriever
# from .filter_retriever import FilterRetriever as FilterRetriever
from .in_memory import InMemoryBM25Retriever as InMemoryBM25Retriever
from .in_memory import InMemoryEmbeddingRetriever as InMemoryEmbeddingRetriever
# from .sentence_window_retriever import SentenceWindowRetriever as SentenceWindowRetriever
# from .indexs import FaissIndex as FaissIndex
# from .indexs import MilvusIndex as MilvusIndex
