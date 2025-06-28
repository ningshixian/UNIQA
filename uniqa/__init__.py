# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# We avoid lazy imports here because:
# - they create potential static type checking issues which are hard to debug
# - they make this module more complicated and hard to maintain
# - they offer minimal performance gains in this case.

import uniqa.logging
# import uniqa.tracing
# from haystack.core.component import component
from uniqa.core.errors import ComponentError, DeserializationError
# from haystack.core.pipeline import AsyncPipeline, Pipeline, PredefinedPipeline
from uniqa.core.serialization import default_from_dict, default_to_dict
# from haystack.core.super_component.super_component import SuperComponent, super_component
from uniqa.dataclasses import Answer, Document, ExtractedAnswer, GeneratedAnswer
# from uniqa.version import __version__

# # Initialize the logging configuration
# # This is a no-op unless `structlog` is installed
# uniqa.logging.configure_logging()

# Same for tracing (no op if `opentelemetry` or `ddtrace` is not installed)
# uniqa.tracing.auto_enable_tracing()

__all__ = [
    "Answer",
    # "AsyncPipeline",
    "ComponentError",
    "DeserializationError",
    "Document",
    "ExtractedAnswer",
    "GeneratedAnswer",
    # "Pipeline",
    # "PredefinedPipeline",
    # "SuperComponent",
    # "super_component",
    # "component",
    "default_from_dict",
    "default_to_dict",
]