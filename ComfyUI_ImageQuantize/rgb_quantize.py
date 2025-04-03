# Add this to custom_nodes folder

import torch
import numpy as np
from PIL import Image
from comfy.utils import common_upscale

class RGBImageQuantize:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE",),
            "colors": ("INT", {"default": 16, "min": 2, "max": 256, "step": 1}),
            "dither": (["none", "floyd-steinberg", "bayer-2", "bayer-4", "bayer-8"], {"default": "none"}),
            }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "quantize"
    CATEGORY = "image/processing"

    def quantize(self, images, colors, dither):
        batch_size, height, width, channels = images.shape
        
        # Make sure we're working with RGB only
        if channels != 3:
            images = images[:, :, :, :3]
        
        result = torch.zeros_like(images)
        
        # Convert to numpy for PIL
        for b in range(batch_size):
            img = images[b].numpy()
            img = (img * 255).astype(np.uint8)
            
            # Convert to PIL
            pil_img = Image.fromarray(img)
            
            # Setup dithering method
            if dither == "none":
                dither_mode = Image.Dither.NONE
            elif dither == "floyd-steinberg":
                dither_mode = Image.Dither.FLOYDSTEINBERG
            else:
                # Handle Bayer dithering
                dither_mode = Image.Dither.NONE
                # Implement custom bayer dithering if needed
            
            # Quantize to the specified number of colors
            pil_quantized = pil_img.quantize(colors=colors, dither=dither_mode).convert("RGB")
            
            # Convert back to torch tensor
            quantized_array = np.array(pil_quantized).astype(np.float32) / 255.0
            result[b] = torch.from_numpy(quantized_array)
            
        return (result,)

# Additional utility function for adding alpha channel
class AddAlphaChannel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE",),
            }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_alpha"
    CATEGORY = "image/processing"

    def add_alpha(self, images):
        batch_size, height, width, channels = images.shape
        
        # Create new tensor with 4 channels
        result = torch.ones((batch_size, height, width, 4), dtype=images.dtype)
        
        # Copy RGB channels
        result[:, :, :, :3] = images[:, :, :, :3]
        
        return (result,)

# Register the nodes
NODE_CLASS_MAPPINGS = {
    "RGBImageQuantize": RGBImageQuantize,
    "AddAlphaChannel": AddAlphaChannel
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RGBImageQuantize": "RGB Image Quantize",
    "AddAlphaChannel": "Add Alpha Channel"
}