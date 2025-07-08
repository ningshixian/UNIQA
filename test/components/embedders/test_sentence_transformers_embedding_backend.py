# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('.'))

from uniqa.components.embedders.backends.sentence_transformers_backend import (
    _SentenceTransformersEmbeddingBackendFactory,
)


embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
    model="infgrad/stella-base-zh-v3-1792d",   # dunzhang/stella-large-zh-v3-1792d
    device="cpu",    
    trust_remote_code=True,
    local_files_only=True,
    truncate_dim=256,
    backend="torch",
)

data = ["sentence1", "sentence2"]
result = embedding_backend.embed(data=data, normalize_embeddings=True)
print(result)
