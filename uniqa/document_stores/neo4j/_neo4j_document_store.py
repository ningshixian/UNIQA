# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import math
import re
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
# import bm25s
# from bm25s.tokenization import Tokenized
import hanlp
from tqdm.auto import tqdm

from uniqa import logging
from uniqa import default_from_dict, default_to_dict, logging
from uniqa.dataclasses import Document
# from uniqa.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from uniqa.document_stores.types import DuplicatePolicy
from uniqa.utils import expit
from uniqa.utils.filters import document_matches_filter

from uniqa import Document
from uniqa.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install neo4j_haystack'") as neo4j_import:
    from neo4j_haystack import Neo4jDocumentStore
    # from neo4j_haystack import Neo4jEmbeddingRetriever, Neo4jDynamicDocumentRetriever

logger = logging.logDog
