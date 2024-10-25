import torch
import gc

class MemoryOptimizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 64
                }),
                "max_height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 64
                }),
                "use_half_precision": ("BOOLEAN", {"default": True}),
                "keep_aspect_ratio": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "optimize_memory"
    CATEGORY = "utils"

    def optimize_memory(self, image, max_width, max_height, use_half_precision, keep_aspect_ratio):
        # Ensure input is a torch tensor
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)

        current_height, current_width = image.shape[-2], image.shape[-1]
        
        # Check if resizing is needed
        if current_width > max_width or current_height > max_height:
            if keep_aspect_ratio:
                # Calculate scaling factors for both dimensions
                width_scale = max_width / current_width
                height_scale = max_height / current_height
                
                # Use the smaller scale to ensure both dimensions fit within limits
                scale = min(width_scale, height_scale)
                
                new_width = int(current_width * scale)
                new_height = int(current_height * scale)
            else:
                new_width = max_width
                new_height = max_height
            
            image = torch.nn.functional.interpolate(image, size=(new_height, new_width), mode='bilinear', align_corners=False)

        # Convert to half precision if requested
        if use_half_precision:
            image = image.half()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Perform garbage collection
        gc.collect()

        return (image,)

    @classmethod
    def IS_CHANGED(s, image, max_width, max_height, use_half_precision, keep_aspect_ratio):
        return float("NaN")
