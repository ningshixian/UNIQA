"""
Model2Vec module
"""

import json
from typing import Any, Dict, List, Literal, Optional
from huggingface_hub.errors import HFValidationError

# Conditional import
try:
    from model2vec import StaticModel

    MODEL2VEC = True
except ImportError:
    MODEL2VEC = False


class Model2VecTextEmbedder:
    """
    Builds vectors using Model2Vec.
    """

    def __init__(self, config, scoring, models):
        self.config = config
        self.scoring = scoring
        self.models = models

        if config:
            # Detect if this is an initialized configuration
            self.initialized = "dimensions" in config

            # Enables optional string tokenization
            self.tokenize = config.get("tokenize")

            # Load model
            self.model = self.load(config.get("path"))

            # Encode batch size - controls underlying model batch size when encoding vectors
            self.encodebatch = config.get("encodebatch", 32)

            # Embeddings instructions
            self.instructions = config.get("instructions")

            # Truncate embeddings to this dimensionality
            self.dimensionality = config.get("dimensionality")

            # Scalar quantization - supports 1-bit through 8-bit quantization
            quantize = config.get("quantize")
            self.qbits = max(min(quantize, 8), 1) if isinstance(quantize, int) and not isinstance(quantize, bool) else None

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.models}
    @staticmethod
    def ismodel(path):
        """
        Checks if path is a Model2Vec model.

        Args:
            path: input path

        Returns:
            True if this is a Model2Vec model, False otherwise
        """

        try:
            # Download file and parse JSON
            from transformers.utils import cached_file
            path = cached_file(path_or_repo_id=path, filename="config.json")
            if path:
                with open(path, encoding="utf-8") as f:
                    config = json.load(f)
                    return config.get("model_type") == "model2vec"

        # Ignore this error - invalid repo or directory
        except Exception as e:
            pass

        return False

    def loadmodel(self, path):
        return StaticModel.from_pretrained(path)

    def run(self, text: str):
        # Additional model arguments
        modelargs = self.config.get("vectors", {})

        # Encode data
        return self.model.encode(text, batch_size=self.encodebatch, **modelargs)