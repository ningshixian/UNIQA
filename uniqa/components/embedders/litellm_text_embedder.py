"""
LiteLLM module
"""

from typing import Any, Dict, List, Literal, Optional
import numpy as np

# Conditional import
try:
    import litellm as api

    LITELLM = True
except ImportError:
    LITELLM = False


class LiteLLMTextEmbedder:
    """
    Builds vectors using an external embeddings API via LiteLLM.
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
        Checks if path is a LiteLLM model.

        Args:
            path: input path

        Returns:
            True if this is a LiteLLM model, False otherwise
        """

        # pylint: disable=W0702
        if isinstance(path, str) and LITELLM:
            debug = api.suppress_debug_info
            try:
                # Suppress debug messages for this test
                api.suppress_debug_info = True
                return api.get_llm_provider(path)
            except:
                return False
            finally:
                # Restore debug info value to original value
                api.suppress_debug_info = debug

        return False

    def loadmodel(self, path):
        return None

    def run(self, text: str):
        # Call external embeddings API using LiteLLM
        # Batching is handled server-side
        response = api.embedding(model=self.config.get("path"), input=text, **self.config.get("vectors", {}))

        # Read response into a NumPy array
        return np.array([x["embedding"] for x in response.data], dtype=np.float32)