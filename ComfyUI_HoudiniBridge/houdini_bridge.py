import torch
import os
from PIL import Image
import numpy as np

class HoudiniBridge:
    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.exr')
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "filename": ("STRING", {
                    "default": "Depth.jpg",
                    "multiline": False
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"
    CATEGORY = "loaders/houdini"

    def validate_path(self, base_path, filename):
        if not os.path.isdir(base_path):
            raise ValueError(f"Invalid base path: {base_path}")
        
        image_path = os.path.join(base_path, filename)
        ext = os.path.splitext(filename)[1].lower()
        
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {ext}. Supported formats: {', '.join(self.SUPPORTED_FORMATS)}")
            
        return image_path

    def load_image(self, base_path, filename):
        """
        Load image from base path and filename with enhanced error handling and proper data type conversion
        """
        try:
            image_path = self.validate_path(base_path, filename)
            
            if not os.path.exists(image_path):
                print(f"Error: Image not found at {image_path}")
                return (torch.zeros((1, 3, 64, 64), dtype=torch.float32),)

            # Load and verify image
            img = Image.open(image_path)
            
            # Convert grayscale to RGB if needed
            if img.mode == 'L':
                img = img.convert('RGB')
            elif img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            # Convert to numpy array with explicit float32 type
            img_np = np.array(img, dtype=np.float32)
            
            # Ensure 3 channels (RGB)
            if len(img_np.shape) == 2:  # Grayscale
                img_np = np.stack([img_np] * 3, axis=-1)
            
            # Normalize to 0-1 range
            if img_np.max() > 1.0:
                img_np = img_np / 255.0
            
            # Ensure proper shape (batch, channels, height, width)
            if len(img_np.shape) == 3:
                img_np = np.transpose(img_np, (2, 0, 1))  # CHW format
                img_np = np.expand_dims(img_np, 0)  # Add batch dimension
            
            # Convert to torch tensor with correct dtype
            img_tensor = torch.from_numpy(img_np).float()
            
            # Ensure proper shape and type
            if img_tensor.shape[1] != 3:
                raise ValueError(f"Expected 3 channels, got {img_tensor.shape[1]}")
            
            print(f"Successfully loaded image. Shape: {img_tensor.shape}, Type: {img_tensor.dtype}")
            return (img_tensor,)
            
        except ValueError as ve:
            print(f"Validation error: {str(ve)}")
            return (torch.zeros((1, 3, 64, 64), dtype=torch.float32),)
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            print(f"Image path: {image_path if 'image_path' in locals() else 'unknown'}")
            return (torch.zeros((1, 3, 64, 64), dtype=torch.float32),)

    @classmethod
    def IS_CHANGED(s, base_path, filename):
        """
        Check if the image file has been modified
        """
        full_path = os.path.join(base_path, filename)
        if os.path.exists(full_path):
            return str(os.path.getmtime(full_path))
        return "0"

NODE_CLASS_MAPPINGS = {
    "HoudiniBridge": HoudiniBridge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HoudiniBridge": "Houdini Render Bridge"
}