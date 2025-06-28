# embedders 模块的任务是 获取文本的向量表示。

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    # "litellm_text_embedder": ["LiteLLMTextEmbedder"],
    # "m2v_text_embedder": ["Model2VecTextEmbedder"],
    "sentence_transformers_document_embedder": ["SentenceTransformersDocumentEmbedder"],
    "sentence_transformers_text_embedder": ["SentenceTransformersTextEmbedder"],
}

if TYPE_CHECKING:
    from .sentence_transformers_document_embedder import (
        SentenceTransformersDocumentEmbedder as SentenceTransformersDocumentEmbedder,
    )
    from .sentence_transformers_text_embedder import (
        SentenceTransformersTextEmbedder as SentenceTransformersTextEmbedder,
    )
    # from .litellm_text_embedder import (
    #     LiteLLMTextEmbedder as LiteLLMTextEmbedder,
    # )
    # from .m2v_text_embedder import (
    #     Model2VecTextEmbedder as Model2VecTextEmbedder,
    # )
else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
