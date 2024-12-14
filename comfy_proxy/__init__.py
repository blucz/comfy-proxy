"""
ComfyProxy - A Python client library for ComfyUI

This module provides classes and utilities for interacting with ComfyUI servers
and managing image generation workflows.
"""

from .comfy import SingleComfy, Comfy
from .workflow import (
    ComfyWorkflow,
    ComfyNode,
    Sizes,
    Size,
    Lora
)

__all__ = [
    'SingleComfy',
    'Comfy',
    'ComfyWorkflow', 
    'ComfyNode',
    'Sizes',
    'Size',
    'Lora'
]
