# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

__all__ = ["ElasticsearchBM25Retriever", "ElasticsearchEmbeddingRetriever"]
_import_structure = {
    "elasticsearch_embedding_retriever": ["ElasticsearchEmbeddingRetriever"], 
    "elasticsearch_bm25_retriever": ["ElasticsearchBM25Retriever"]
}

if TYPE_CHECKING:
    from ._elasticsearch_bm25_retriever import ElasticsearchBM25Retriever
    from ._elasticsearch_embedding_retriever import ElasticsearchEmbeddingRetriever

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)