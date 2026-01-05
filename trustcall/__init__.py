"""Utilities for validated tool calling and extraction with retries using LLMs.

This module provides functionality for creating extractors that can generate,
validate, and correct structured outputs from language models. It supports
patch-based extraction for efficient and accurate updates to existing schemas.
"""

from trustcall._base import (
    ExtractionInputs,
    ExtractionOutputs,
    MultiObjectExtractionOutputs,
    create_extractor,
    create_multi_object_extractor,
)

__all__ = [
    "create_extractor",
    "create_multi_object_extractor",
    "ExtractionInputs",
    "ExtractionOutputs",
    "MultiObjectExtractionOutputs",
]
