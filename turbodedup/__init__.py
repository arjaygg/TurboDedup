"""
TurboDedup - High-Performance File Deduplication Scanner

A comprehensive file deduplication solution with GPU acceleration, smart caching,
and similarity detection capabilities.
"""

__version__ = "4.0.0"
__author__ = "TurboDedup Team"
__email__ = "info@turbodedup.dev"
__license__ = "MIT"

from .core import cache_manager, similarity_detector, gpu_accelerator

__all__ = [
    "cache_manager",
    "similarity_detector", 
    "gpu_accelerator",
]