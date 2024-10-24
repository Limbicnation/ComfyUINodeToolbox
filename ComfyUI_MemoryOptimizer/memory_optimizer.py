import torch
import gc

class MemoryOptimizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_resolution": ("INT", {
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

    def optimize_memory(self, image, max_resolution, use_half_precision, keep_aspect_ratio):
        # Ensure input is a torch tensor
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)

        # Resize image if necessary
        if image.shape[-1] > max_resolution or image.shape[-2] > max_resolution:
            if keep_aspect_ratio:
                aspect_ratio = image.shape[-2] / image.shape[-1]
                if aspect_ratio > 1:
                    new_height = max_resolution
                    new_width = int(max_resolution / aspect_ratio)
                else:
                    new_width = max_resolution
                    new_height = int(max_resolution * aspect_ratio)
            else:
                new_height = max_resolution
                new_width = max_resolution
            
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
    def IS_CHANGED(s, image, max_resolution, use_half_precision, keep_aspect_ratio):
        return float("NaN")
