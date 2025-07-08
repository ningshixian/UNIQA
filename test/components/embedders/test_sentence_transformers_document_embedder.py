# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('.'))

import random
from unittest.mock import MagicMock, patch

import pytest
import torch

from uniqa import Document
from uniqa.components.embedders.sentence_transformers_document_embedder import SentenceTransformersDocumentEmbedder
from uniqa.utils import ComponentDevice
print(ComponentDevice.resolve_device(None))  # MPS

documents = [Document(content=f"document number {i}") for i in range(5)]

embedder = SentenceTransformersDocumentEmbedder(
    model="infgrad/stella-base-zh-v3-1792d",   # dunzhang/stella-large-zh-v3-1792d
    device=ComponentDevice.resolve_device(None),    # ComponentDevice.from_str("cuda:0")
    batch_size=32, 
    progress_bar=True,
    normalize_embeddings=True,
    precision="float32",
    trust_remote_code=True,
    # meta_fields_to_embed=[], 
    # embedding_separator=" | ",
    # prefix="prefix", 
    # suffix="suffix", 
)
embedder.warm_up()
embedder.warm_up()
result = embedder.run(documents)
print(result["documents"])
# [Document(id=..., content: '...', embedding: vector of size 1792), ...]

assert isinstance(result["documents"], list)
assert len(result["documents"]) == len(documents)
for doc in result["documents"]:
    assert isinstance(doc, Document)
    assert isinstance(doc.embedding, list)
    assert isinstance(doc.embedding[0], float)

