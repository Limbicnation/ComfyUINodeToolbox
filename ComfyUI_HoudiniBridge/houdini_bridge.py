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
                    "default": "/media/gero/Qsync_Ubuntu/Qsync/55_Houdini_Projects_Linux/1_3D/Houdini/1_Scenes/StableHoudini_Linux/Render/Temp",
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

            # Load image
            i = Image.open(image_path)
            
            # Convert to RGB if it's not already
            if i.mode != 'RGB':
                # For depth maps, convert to RGB by duplicating the channel
                if i.mode in ['L', 'I']:
                    i = Image.merge('RGB', (i, i, i))
                else:
                    i = i.convert('RGB')
            
            # Convert to numpy array and normalize
            i = np.array(i).astype(np.float32) / 255.0
            
            # Convert to torch tensor
            i = torch.from_numpy(i)[None,]
            
            # Rearrange dimensions if needed
            if len(i.shape) == 4:
                i = i.permute(0, 3, 1, 2)
            
            return (i,)
            
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            print(f"Image path: {image_path}")
            print(f"Image mode: {getattr(i, 'mode', 'Unknown')}")
            print(f"Image size: {getattr(i, 'size', 'Unknown')}")
            return (torch.zeros((1, 3, 64, 64)),)
