"""
ComfyUI Image Reloader - A Dynamic Image Loading Node
"""

import os
import sys

# Add the current directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from nodes import DynamicImageLoader

NODE_CLASS_MAPPINGS = {
    "DynamicImageLoader": DynamicImageLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamicImageLoader": "🔄 Dynamic Image Loader"
}

__version__ = "1.0.0"