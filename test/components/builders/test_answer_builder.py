# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('.'))
import logging

import pytest

from uniqa import Document, GeneratedAnswer
from uniqa.components.builders.answer_builder import AnswerBuilder
from uniqa.dataclasses.chat_message import ChatMessage

"""pytest -svx test/compenents/builders/test_answer_builder.py"""


builder = AnswerBuilder(pattern="Answer: (.*)")
documents = [
    Document(content="test doc 1"),
    Document(content="test doc 2"),
    Document(content="test doc 3")
]   # 参考文档
answers = builder.run(
    query="What's the answer?", 
    replies=["This is an argument. Answer: This is the answer.[2][3]"],
    meta=[{}],
    documents=documents,
    # reference_pattern="\\[(\\d+)\\]", # 用于解析documents引用的正则表达式。如果未指定，则不进行解析，所有文档都会被引用。
)
print(answers['answers'])

# Turns the output of a Generator into `GeneratedAnswer` objects using regular expressions.

