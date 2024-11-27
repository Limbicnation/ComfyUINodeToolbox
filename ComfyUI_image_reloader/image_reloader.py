import os
import torch
import numpy as np
from PIL import Image
from typing import Tuple

class DynamicImageLoader:
    """Dynamic image loader with reload functionality for ComfyUI"""
    
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

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"
    CATEGORY = "image"
    
    def __init__(self):
        self.last_mtime = 0
    
    def load_image(self, image_directory: str, image_name: str, reload: int) -> torch.Tensor:
        """Load and process image into correct tensor format for ComfyUI"""
        try:
            full_path = os.path.join(image_directory, image_name)
            
            # Validate path and file
            if not os.path.exists(full_path) or not os.path.isfile(full_path):
                print(f"[Image Reloader] Error: Invalid path or file: {full_path}")
                return self._get_empty_image()
            
            # Check modification time
            try:
                current_mtime = os.path.getmtime(full_path)
                if current_mtime == self.last_mtime and reload == 0:
                    return self._get_empty_image()
                self.last_mtime = current_mtime
            except OSError as e:
                print(f"[Image Reloader] Error accessing file: {str(e)}")
                return self._get_empty_image()
            
            # Load and process image
            try:
                # Open and validate image
                img = Image.open(full_path)
                if img.size[0] == 0 or img.size[1] == 0:
                    raise ValueError(f"Invalid image dimensions: {img.size}")
                
                # Convert to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to numpy array
                img_np = np.array(img, dtype=np.float32)
                
                # Validate array dimensions
                if img_np.ndim != 3 or img_np.shape[2] != 3:
                    raise ValueError(f"Invalid array shape: {img_np.shape}")
                
                # Convert to tensor with correct dimensions for ComfyUI [B,C,H,W]
                img_tensor = torch.from_numpy(img_np.copy()).float() / 255.0  # Use copy() to ensure memory contiguity
                
                # Ensure channel dimension is second
                if img_tensor.shape[-1] == 3:
                    img_tensor = img_tensor.permute(2, 0, 1)
                
                # Add batch dimension if not present
                if img_tensor.dim() == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                
                # Validate final tensor shape
                if img_tensor.dim() != 4 or img_tensor.shape[1] != 3:
                    raise ValueError(f"Invalid tensor dimensions: {img_tensor.shape}")
                
                print(f"[Image Reloader] Loaded image shape: {img_tensor.shape}")
                return img_tensor
                
            except Exception as e:
                print(f"[Image Reloader] Error processing image: {str(e)}")
                return self._get_empty_image()
                
        except Exception as e:
            print(f"[Image Reloader] Unexpected error: {str(e)}")
            return self._get_empty_image()
    
    def _get_empty_image(self) -> torch.Tensor:
        """Return empty image tensor in correct ComfyUI format [B,C,H,W]"""
        # Default size matching typical image dimensions
        return torch.zeros((1, 3, 512, 512), dtype=torch.float32)
    
    @classmethod
    def IS_CHANGED(cls, image_directory: str, image_name: str, reload: int) -> str:
        """Check if image needs reloading"""
        path = os.path.join(image_directory, image_name)
        try:
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                return f"{mtime}_{reload}"
        except Exception:
            pass
        return "0"