# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('.'))

import logging

import pytest

from uniqa import Document
from uniqa.dataclasses import ByteStream, SparseEmbedding
from uniqa.components.preprocessors import DocumentCleaner


doc = Document(content="This   is  a  document  to  clean. \n\n\n substring to remove")

cleaner = DocumentCleaner(
    remove_empty_lines = True,
    remove_extra_whitespaces = True,
    remove_repeated_substrings = False,
    keep_id = False,
    remove_regex=r"\s\s+", 
    remove_substrings = ["substring to remove"]
)
result = cleaner.run(documents=[doc])
print(result["documents"])

assert result["documents"][0].content == "This is a document to clean."

