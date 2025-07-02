# preprocessors 模块的任务是 对文本进行预处理。

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    # "csv_document_cleaner": ["CSVDocumentCleaner"],
    # "csv_document_splitter": ["CSVDocumentSplitter"],
    "document_cleaner": ["DocumentCleaner"],
    # "document_preprocessor": ["DocumentPreprocessor"],
    # "document_splitter": ["DocumentSplitter"],
    "recursive_splitter": ["RecursiveDocumentSplitter"],
    "text_cleaner": ["TextCleaner"],
    "document_splitter_zh": ["ChineseDocumentSpliter"],
}

if TYPE_CHECKING:
    # from .csv_document_cleaner import CSVDocumentCleaner as CSVDocumentCleaner
    # from .csv_document_splitter import CSVDocumentSplitter as CSVDocumentSplitter
    from .document_cleaner import DocumentCleaner as DocumentCleaner
    # from .document_preprocessor import DocumentPreprocessor as DocumentPreprocessor
    # from .document_splitter import DocumentSplitter as DocumentSplitter
    from .recursive_splitter import RecursiveDocumentSplitter as RecursiveDocumentSplitter
    from .text_cleaner import TextCleaner as TextCleaner
    from .document_splitter_zh import ChineseDocumentSpliter as ChineseDocumentSpliter

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
