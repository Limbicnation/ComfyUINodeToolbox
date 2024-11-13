import torch
import os
from PIL import Image
import numpy as np

class HoudiniBridge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_path": ("STRING", {
                    "default": "",  # Empty default, user must set their own path
                    "multiline": False
                }),
                "filename": ("STRING", {
                    "default": "render.jpg",  # Generic default filename
                    "multiline": False
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"
    CATEGORY = "loaders/houdini"

    def load_image(self, base_path, filename):
        """
        Load image from base path and filename
        """
        try:
            # Combine path and filename
            image_path = os.path.join(base_path, filename)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}")
                return (torch.zeros((1, 3, 64, 64)),)

            # Load the image in RGB mode
            img = Image.open(image_path).convert('RGB')
            
            # Convert to numpy array and ensure float32
            img_np = np.array(img, dtype=np.float32)
            
            # Normalize to 0-1 range
            img_np = img_np / 255.0
            
            # Reshape to match expected format (batch, channels, height, width)
            img_np = img_np.transpose(2, 0, 1)
            img_np = img_np[None, ...]
            
            # Convert to torch tensor
            img_tensor = torch.from_numpy(img_np)
            
            print(f"Loaded image shape: {img_tensor.shape}")
            return (img_tensor,)
            
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            print(f"Image path: {image_path}")
            return (torch.zeros((1, 3, 64, 64)),)
