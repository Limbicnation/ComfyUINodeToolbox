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
        Load image from base path and filename with ComfyUI-compatible output format
        """
        try:
            image_path = self.validate_path(base_path, filename)
            
            if not os.path.exists(image_path):
                print(f"Error: Image not found at {image_path}")
                # Return black image in correct format (B,H,W,C)
                return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

            # Load and verify image
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array with explicit float32 type
            img_np = np.array(img, dtype=np.float32)
            
            # Normalize to 0-1 range
            img_np = img_np / 255.0
            
            # Add batch dimension if needed (B,H,W,C)
            if len(img_np.shape) == 3:
                img_np = np.expand_dims(img_np, 0)
            
            # Convert to torch tensor
            img_tensor = torch.from_numpy(img_np)
            
            print(f"Successfully loaded image. Shape: {img_tensor.shape}, Type: {img_tensor.dtype}")
            return (img_tensor,)
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            # Return black image in correct format (B,H,W,C)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

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