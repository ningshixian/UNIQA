# writers 模块定义了 DocumentWriter 类，用于将文档以指定去重策略写入 DocumentStore，
# 并支持同步与异步操作。

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {"document_writer": ["DocumentWriter"]}

if TYPE_CHECKING:
    from .document_writer import DocumentWriter as DocumentWriter

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
