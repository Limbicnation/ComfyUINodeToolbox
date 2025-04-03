import torch
import numpy as np
from PIL import Image

class RGBQuantizeNode:
    """
    A node that quantizes images to reduce the number of colors by RGB value.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "r_levels": ("INT", {"default": 8, "min": 2, "max": 256, "step": 1}),
                "g_levels": ("INT", {"default": 8, "min": 2, "max": 256, "step": 1}),
                "b_levels": ("INT", {"default": 8, "min": 2, "max": 256, "step": 1}),
                "dither": (["NONE", "FLOYDSTEINBERG"], {"default": "NONE"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("quantized_image", "palette_image")
    FUNCTION = "quantize_image"
    CATEGORY = "image/processing"

    def quantize_image(self, image, r_levels, g_levels, b_levels, dither):
        # Convert from tensor to numpy
        i = 255. * image.cpu().numpy()
        img_np = np.clip(i, 0, 255).astype(np.uint8)[0]
        
        # Create RGB color levels
        r_steps = 255 // (r_levels - 1)
        g_steps = 255 // (g_levels - 1)
        b_steps = 255 // (b_levels - 1)
        
        # Create the quantization function
        if dither == "NONE":
            # Simple quantization without dithering
            r_quant = np.round(img_np[:, :, 0] / r_steps) * r_steps
            g_quant = np.round(img_np[:, :, 1] / g_steps) * g_steps
            b_quant = np.round(img_np[:, :, 2] / b_steps) * b_steps
            
            quant_img = np.stack([r_quant, g_quant, b_quant], axis=-1).astype(np.uint8)
        else:
            # For dithering, convert to PIL and use its quantize method
            img_pil = Image.fromarray(img_np)
            
            # Generate a custom palette
            palette = []
            for r in range(r_levels):
                r_val = min(255, r * r_steps)
                for g in range(g_levels):
                    g_val = min(255, g * g_steps)
                    for b in range(b_levels):
                        b_val = min(255, b * b_steps)
                        palette.extend([r_val, g_val, b_val])
            
            # Create a new palette image and set the palette
            palette_img = Image.new('P', (1, 1))
            palette_img.putpalette(palette + [0] * (768 - len(palette)))
            
            # Quantize using Floyd-Steinberg dithering
            quant_img_pil = img_pil.quantize(palette=palette_img, dither=Image.FLOYDSTEINBERG)
            quant_img_pil = quant_img_pil.convert('RGB')
            quant_img = np.array(quant_img_pil)
        
        # Generate palette visualization
        total_colors = r_levels * g_levels * b_levels
        palette_size = min(256, total_colors)  # Limit to 256 for visualization
        
        # Calculate ideal palette image dimensions
        width = int(np.sqrt(palette_size))
        height = (palette_size + width - 1) // width
        
        palette_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        idx = 0
        for r in range(r_levels):
            r_val = min(255, r * r_steps)
            for g in range(g_levels):
                g_val = min(255, g * g_steps)
                for b in range(b_levels):
                    if idx >= palette_size:
                        break
                    b_val = min(255, b * b_steps)
                    
                    y, x = divmod(idx, width)
                    if y < height:  # Safety check
                        palette_img[y, x] = [r_val, g_val, b_val]
                    idx += 1
                if idx >= palette_size:
                    break
            if idx >= palette_size:
                break
        
        # Convert back to tensor
        quantized_tensor = torch.from_numpy(quant_img.astype(np.float32) / 255.0).unsqueeze(0)
        palette_tensor = torch.from_numpy(palette_img.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (quantized_tensor, palette_tensor)

NODE_CLASS_MAPPINGS = {
    "RGBQuantize": RGBQuantizeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RGBQuantize": "RGB Quantize"
}