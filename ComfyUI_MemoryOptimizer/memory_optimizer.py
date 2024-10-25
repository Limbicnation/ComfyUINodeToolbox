import torch
import gc

class MemoryOptimizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64
                }),
                "max_height": ("INT", {
                    "default": 576,
                    "min": 64,
                    "max": 8192,
                    "step": 64
                }),
                "force_square": ("BOOLEAN", {"default": False}),
                "keep_aspect_ratio": ("BOOLEAN", {"default": True}),
                "use_half_precision": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "optimize_memory"
    CATEGORY = "utils"

    def optimize_memory(self, image, max_width, max_height, force_square, keep_aspect_ratio, use_half_precision):
        # Ensure input is a torch tensor
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)

        # Get current dimensions
        batch_size, channels, current_height, current_width = image.shape

        # Handle channel expansion first if needed
        if channels == 1:
            image = image.repeat(1, 3, 1, 1)

        # Handle square image requirement
        if force_square:
            max_size = min(max_width, max_height)
            max_width = max_size
            max_height = max_size

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
            
            # Perform the resize operation
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
    def IS_CHANGED(s, image, max_width, max_height, force_square, keep_aspect_ratio, use_half_precision):
        return float("NaN")
