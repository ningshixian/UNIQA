# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-public-methods

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('.'))

from unittest.mock import Mock, patch

import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from uniqa.components.generators.hugging_face_local import HuggingFaceLocalGenerator, StopWordsCriteria
from uniqa.utils import ComponentDevice
# print(ComponentDevice.resolve_device(None).to_hf())


generator = HuggingFaceLocalGenerator(
    model="Qwen/Qwen3-0.6B",
    task="text-generation",
    device=ComponentDevice.from_str("cpu"),  #
    # device=ComponentDevice.resolve_device(None),
    generation_kwargs={"max_new_tokens": 512, "temperature": 0.9})

# generator.huggingface_pipeline_kwargs["torch_dtype"] = torch.float16   # RuntimeError: isin_Tensor_Tensor_out only works on floating types on MPS for pre MacOS_14_0. Received dtype: Long
# generator.huggingface_pipeline_kwargs["torch_dtype"] = "auto"  # TypeError: MPS BFloat16 is only supported on MacOS 14 or newer

generator.warm_up()
result = generator.run("Who is the best American actor?")
print(result["replies"])
# {'replies': ['John Cusack']}

result = generator.run(prompt="")
print(result["replies"])


# You must use the `apply_chat_template` method to add the generation prompt to properly include the instruction
# tokens in the prompt. Otherwise, the model will not generate the expected output.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
messages = [{"role": "user", "content": "Please repeat the phrase 'climate change' and nothing else"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

result = generator.run(prompt=prompt)
print(result["replies"])
