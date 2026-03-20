from __future__ import annotations

"""
Compatibility wrapper.

Core prompt construction lives in `prompt_builder/builder.py` so it can be shared across training/inference/mobile.
This module provides a stable import path under `inference/` as requested.
"""

from prompt_builder.builder import DEFAULT_SYSTEM_PREFIX, PromptBuilder

__all__ = ["PromptBuilder", "DEFAULT_SYSTEM_PREFIX"]

