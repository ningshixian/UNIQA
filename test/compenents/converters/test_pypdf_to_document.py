# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath("."))

import logging
from unittest.mock import patch, Mock

import pytest

from uniqa import Document
from uniqa.components.converters.pypdf import PyPDFToDocument, PyPDFExtractionMode
from uniqa.components.preprocessors import DocumentSplitter
from uniqa.dataclasses import ByteStream


test_files_path = "examples/test_documents/天池初赛训练数据集.pdf"

converter = PyPDFToDocument()
paths = [test_files_path]
with open(test_files_path, "rb") as f:
    paths.append(ByteStream(f.read()))

output = converter.run(sources=paths, meta={"language": "it"})
docs = output["documents"]
print(docs[0].content[:100])
assert len(docs) == 2
assert docs[0].content.count("\f") == 353


converter = PyPDFToDocument(extraction_mode=PyPDFExtractionMode.LAYOUT)
sources = [test_files_path]
pdf_doc = converter.run(sources=sources)
splitter = DocumentSplitter(split_length=1, split_by="passage")
docs = splitter.run(pdf_doc["documents"])
print(docs['documents'][:10])
