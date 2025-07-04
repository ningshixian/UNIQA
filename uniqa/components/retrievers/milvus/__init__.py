# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {"milvus_embedding_retriever": ["MilvusEmbeddingRetriever", "MilvusHybridRetriever", "MilvusSparseEmbeddingRetriever"]}
__all__ = ["MilvusEmbeddingRetriever", "MilvusHybridRetriever", "MilvusSparseEmbeddingRetriever"]

if TYPE_CHECKING:
    from .milvus_embedding_retriever import MilvusEmbeddingRetriever as MilvusEmbeddingRetriever
    from .milvus_embedding_retriever import MilvusHybridRetriever as MilvusHybridRetriever
    from .milvus_embedding_retriever import MilvusSparseEmbeddingRetriever as MilvusSparseEmbeddingRetriever

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)