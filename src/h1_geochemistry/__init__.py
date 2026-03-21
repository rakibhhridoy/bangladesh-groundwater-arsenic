"""
Hypothesis 1: Anthropogenically-Modified Reductive Dissolution Framework
Testing groundwater arsenic dynamics using rigorous statistical analysis.
"""

__version__ = "1.0.0"
__author__ = "Groundwater Arsenic Research Team"

from .data_loader import load_and_prepare_data
from .config import *

__all__ = [
    'load_and_prepare_data',
    'PROJECT_ROOT',
    'DATA_DIR',
    'OUTPUT_DIR',
]
