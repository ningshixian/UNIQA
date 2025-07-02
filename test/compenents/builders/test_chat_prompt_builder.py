# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('.'))

from typing import Any, Dict, List, Optional
from jinja2 import TemplateSyntaxError
import arrow
import logging
import pytest

from uniqa.components.builders.chat_prompt_builder import ChatPromptBuilder
# from uniqa import component
# from uniqa.core.pipeline.pipeline import Pipeline
from uniqa.dataclasses.chat_message import ChatMessage
from uniqa.dataclasses.document import Document


prompt_builder = ChatPromptBuilder()
messages = [
    ChatMessage.from_system("Write your response in this language:{{language}}"),
    ChatMessage.from_user("Tell me about {{location}}"),
]
language = "French"
location = "Berlin"
result = prompt_builder.run(
    template_variables={"language": language, "location": location}, 
    template=messages
)
print(result)

