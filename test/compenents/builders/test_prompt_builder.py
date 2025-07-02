# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('.'))

from typing import Any, Dict, List, Optional
from unittest.mock import patch

import arrow
import logging
import pytest
from jinja2 import TemplateSyntaxError

# from uniqa import component
from uniqa.components.builders.prompt_builder import PromptBuilder
# from uniqa.core.pipeline.pipeline import Pipeline
from uniqa.dataclasses.document import Document


template = "Translate the following context to {{ target_language }}. Context: {{ snippet }}; Translation:"
builder = PromptBuilder(template=template)
result = builder.run(target_language="spanish", snippet="I can't speak spanish.")
print(result['prompt'])
