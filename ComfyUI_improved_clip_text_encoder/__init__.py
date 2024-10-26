"""
Flux-enhanced CLIP Text Encoder for ComfyUI
"""

from .improved_clip_text_encoder import CLIPTextEncodeFlux

# Register the node class
NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeFlux": CLIPTextEncodeFlux
}

# Register the display name
NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeFlux": "CLIP Text Encode (Flux)"
}

# Make these mappings available to ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
