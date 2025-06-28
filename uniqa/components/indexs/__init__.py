# indexs 模块的任务是 索引文本。

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "faiss_index": ["FaissIndex"], 
    "milvus_index": ["MilvusIndex"], 
    # "annoy_index": ["AnnoyIndex"], 
}

if TYPE_CHECKING:
    from .faiss_index import FaissIndex as FaissIndex
    from .milvus_index import MilvusIndex as MilvusIndex
    # from .annoy_index import AnnoyIndex as AnnoyIndex

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)