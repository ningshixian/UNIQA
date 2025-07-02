# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('.'))

from datetime import datetime
import pytest
from openai import OpenAIError
from openai.types.chat import ChatCompletionChunk, chat_completion_chunk
from unittest.mock import MagicMock, patch

from uniqa.components.generators import OpenAIGenerator
# from uniqa.components.generators.utils import print_streaming_chunk
# from uniqa.dataclasses import StreamingChunk
# from uniqa.utils.auth import Secret


import os
# https://lpai-llm.lixiang.com/statistics
lpai_token = "ak-infer-ZW50ZXJwcmlzLXNtYXJ0YnVzaW5lc3M6ZnVmcGt6OmZ1ZnBrei1kZWZhdWx0Om5pbmdzaGl4aWFuQGxpeGlhbmcuY29tOmluZmVy_Yjk2NWQ4MmUtMjdmMC00OWVmLWIxZDktMDkzYTc5NjMzYjRl"
os.environ["OPENAI_API_KEY"] = lpai_token

base_url = "https://lpai-llm.lixiang.com/inference/deepseek-ai/deepseek-v3/v1"
model = "deepseek-ai__deepseek-v3-0324"

llm = OpenAIGenerator(
    api_base_url=base_url,
    model=model,
    # api_key=os.environ.get('OPENAI_API_KEY'),
)

prompt = "中国的首都是哪里？"
replies = llm.run(
    prompt,
    system_prompt=None, 
    generation_kwargs = {
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stream": False
    }
)["replies"]
print(replies)
