import os
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Union
import time

class DynamicImageLoader:
    """Dynamic image loader with reload functionality"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_directory": ("STRING", {
                    "default": "ComfyUI/input",
                    "multiline": False,
                }),
                "image_name": ("STRING", {
                    "default": "image.png",
                    "multiline": False,
                }),
                "reload": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999999,
                    "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "image"
    
    def __init__(self):
        self.last_mtime = 0
    
    def load_image(self, image_directory: str, image_name: str, reload: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and process image"""
        try:
            full_path = os.path.join(image_directory, image_name)
            
            # Validate path
            if not os.path.exists(full_path):
                print(f"[Image Reloader] Error: Image not found at {full_path}")
                return self._get_empty_image()
            
            # Check if file was modified
            current_mtime = os.path.getmtime(full_path)
            if current_mtime == self.last_mtime and reload == 0:
                return self._get_empty_image()
            
            self.last_mtime = current_mtime
            
            # Load image
            img = Image.open(full_path)
            
            # Convert to RGB/RGBA if needed
            if img.mode not in ['RGB', 'RGBA']:
                img = img.convert('RGB')
            
            # Convert to numpy array
            img_np = np.array(img)
            
            # Handle alpha channel/mask
            mask = None
            if img.mode == 'RGBA':
                mask = img_np[:, :, 3]
                mask = torch.from_numpy(mask).float() / 255.0
                img_np = img_np[:, :, :3]
            
            # Convert to tensor [B, C, H, W]
            img_tensor = torch.from_numpy(img_np).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            
            print(f"[Image Reloader] Loaded: {image_name}")
            return (img_tensor, mask if mask is not None else torch.zeros((1,)))
            
        except Exception as e:
            print(f"[Image Reloader] Error: {str(e)}")
            return self._get_empty_image()
    
    def _get_empty_image(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return empty image and mask tensors"""
        return (
            torch.zeros((1, 3, 64, 64), dtype=torch.float32),
            torch.zeros((1,), dtype=torch.float32)
        )
    
    @classmethod
    def IS_CHANGED(cls, image_directory: str, image_name: str, reload: int) -> str:
        """Check if image needs reloading"""
        path = os.path.join(image_directory, image_name)
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            return f"{mtime}_{reload}"
        return "0"