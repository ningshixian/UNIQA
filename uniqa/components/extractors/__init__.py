# extractors 模块的任务是 抽取文档中的实体。

import sys
from typing import TYPE_CHECKING

# from lazy_imports import LazyImporter

_import_structure = {
    # "llm_metadata_extractor": ["LLMMetadataExtractor"],
    "named_entity_extractor": ["NamedEntityAnnotation", "NamedEntityExtractor", "NamedEntityExtractorBackend"],
}

# from .llm_metadata_extractor import LLMMetadataExtractor as LLMMetadataExtractor
from .named_entity_extractor import NamedEntityAnnotation as NamedEntityAnnotation
from .named_entity_extractor import NamedEntityExtractor as NamedEntityExtractor
from .named_entity_extractor import NamedEntityExtractorBackend as NamedEntityExtractorBackend
