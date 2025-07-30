"""
Router imports
"""
import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "search": ["search"],
    "entity": ["entity_extract"],
    "similarity": ["similarity"],
    "faq_search_engine": ["predict4cc"],
    "update_api": ["full_update", "incremental_update"],
}

from . import search
from . import entity
from . import similarity
from . import faq_search_engine
from . import update_api
