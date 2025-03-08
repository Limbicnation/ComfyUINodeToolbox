"""
Random Seed Generator Node for ComfyUI
---------------------------------------

A utility node that provides consistent or random seed generation
for stable diffusion workflows. Supports both fixed and random modes
with cross-library synchronization.

Usage:
    1. Add the "Random Seed Generator" node to your workflow
    2. Connect it to nodes that require seed values (samplers, etc.)
    3. Choose between random or fixed mode as needed
"""

__version__ = "1.1.0"
__author__ = "Your Name"

from .random_seed_generator import RandomSeedGeneratorNode

# ComfyUI Node Registration
NODE_CLASS_MAPPINGS = {
    "RandomSeedGenerator": RandomSeedGeneratorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomSeedGenerator": "Random Seed Generator"
}

# Module exports
__all__ = [
    'NODE_CLASS_MAPPINGS', 
    'NODE_DISPLAY_NAME_MAPPINGS',
    'RandomSeedGeneratorNode'
]