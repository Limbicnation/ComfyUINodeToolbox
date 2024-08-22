import torch

class CheckDimensionsNode:
    CATEGORY = "Image/Utilities"

    @classmethod    
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "default_width": ("INT", {"default": 512, "min": 1}),
                "default_height": ("INT", {"default": 512, "min": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "check_dimensions"

    def check_dimensions(self, image, default_width=512, default_height=512):
        # Get image dimensions
        img_height, img_width = image.shape[2:4]
        
        # Check if dimensions are valid
        if img_height <= 0 or img_width <= 0:
            print("Invalid dimensions detected. Returning placeholder image.")
            return torch.zeros((1, 3, default_height, default_width))
        
        return image
