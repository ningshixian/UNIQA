# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath("."))

import pytest

from uniqa.dataclasses import ByteStream
from uniqa.components.converters.txt import TextFileToDocument


test_files_path = "examples/test_documents/sample_doc_1.txt"

bytestream = ByteStream.from_file_path(test_files_path)
bytestream.meta["file_path"] = str(test_files_path)
bytestream.meta["key"] = "value"
files = [test_files_path, bytestream]
converter = TextFileToDocument(store_full_path=False)
output = converter.run(sources=files)
docs = output["documents"]
assert len(docs) == 2
print(docs[0].content)
print(docs[1].content)

