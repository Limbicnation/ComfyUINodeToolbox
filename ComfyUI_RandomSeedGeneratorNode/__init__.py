"""
Advanced Seed Generator Node for ComfyUI
-----------------------------------------

An enhanced utility node that provides comprehensive seed generation
for stable diffusion workflows. Supports multiple modes including
fixed, random, increment, and decrement with robust error handling.

Features:
    - Multiple seed generation modes (fixed, random, increment, decrement)
    - Cross-library synchronization (Python, NumPy, PyTorch)
    - Comprehensive error handling and input validation
    - Configurable logging for debugging
    - Persistent state management across executions
    - CUDA support with deterministic mode options

Usage:
    1. Add the "Advanced Seed Generator" node to your workflow
    2. Connect it to nodes that require seed values (samplers, etc.)
    3. Choose your preferred generation mode
    4. Configure synchronization and deterministic options as needed
"""

__version__ = "2.0.0"
__author__ = "Enhanced by Claude Code"

from .random_seed_generator import AdvancedSeedGenerator, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Re-export the mappings from the main module
# This ensures consistency and single source of truth

# Module exports
__all__ = [
    'NODE_CLASS_MAPPINGS', 
    'NODE_DISPLAY_NAME_MAPPINGS',
    'AdvancedSeedGenerator'
]