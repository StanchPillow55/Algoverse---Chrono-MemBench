"""
Chrono-SAE: Sparse Autoencoder for tracking and steering memory features in LLMs.

A Two-Phase Framework for Tracking and Steering Memory Features in Large Language Models.

This module is part of the algoverse.chrono namespace package.
"""

__version__ = "0.1.0"
__author__ = "Chrono-MemBench Team"

from . import models
from . import training  
from . import analysis
from . import utils
from . import config

# Import main classes for easy access
from .model import ChronoSAE, ChronoSAEConfig, TemporalDropoutGate, create_chrono_sae

__all__ = [
    "models", "training", "analysis", "utils", "config",
    "ChronoSAE", "ChronoSAEConfig", "TemporalDropoutGate", "create_chrono_sae"
]
