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

from uniqa.components.embedders.sentence_transformers_text_embedder import SentenceTransformersTextEmbedder
from uniqa.utils import ComponentDevice


text = "a nice text to embed"

embedder = SentenceTransformersTextEmbedder(
    model="infgrad/stella-base-zh-v3-1792d",   # dunzhang/stella-large-zh-v3-1792d
    device=ComponentDevice.resolve_device(None),    # ComponentDevice.from_str("cuda:0")
    tokenizer_kwargs={"model_max_length": 512},
    batch_size=32, 
    progress_bar=True,
    normalize_embeddings=True,
    precision="float32",
    trust_remote_code=True,
    # prefix="prefix", 
    # suffix="suffix", 
)
embedder.embedding_backend = MagicMock()
embedder.embedding_backend.embed = lambda x, **kwargs: [
    [random.random() for _ in range(16)] for _ in range(len(x))
]

embedder.warm_up()
embedder.warm_up()
result = embedder.run(text=text)
embedding = result["embedding"]
print(embedding)
# [Document(id=..., content: '...', embedding: vector of size 1792), ...]

assert isinstance(embedding, list)
assert all(isinstance(el, float) for el in embedding)
