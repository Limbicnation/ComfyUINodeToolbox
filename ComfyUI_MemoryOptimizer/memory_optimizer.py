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
                "force_cpu_offload": ("BOOLEAN", {"default": False}),
                "aggressive_cleanup": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "optimize_memory"
    CATEGORY = "utils"

    def optimize_memory(self, image, max_width, max_height, force_square, keep_aspect_ratio, 
                       use_half_precision, force_cpu_offload, aggressive_cleanup):
        try:
            # Force garbage collection before processing
            if aggressive_cleanup:
                torch.cuda.empty_cache()
                gc.collect()

            # Move to CPU if requested
            if force_cpu_offload and image.device.type == 'cuda':
                image = image.cpu()

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

                # Move to CPU for resize if memory constrained
                if force_cpu_offload:
                    image = image.cpu()
                
                # Perform the resize operation
                image = torch.nn.functional.interpolate(image, size=(new_height, new_width), 
                                                      mode='bilinear', align_corners=False)

            # Convert to half precision if requested
            if use_half_precision:
                image = image.half()

            # Aggressive cleanup after processing
            if aggressive_cleanup:
                torch.cuda.empty_cache()
                gc.collect()

            return (image,)

        except torch.cuda.OutOfMemoryError:
            # Emergency cleanup on OOM
            torch.cuda.empty_cache()
            gc.collect()
            
            if not force_cpu_offload:
                # Retry with CPU offloading
                return self.optimize_memory(image, max_width, max_height, force_square, 
                                         keep_aspect_ratio, use_half_precision, 
                                         True, True)
            else:
                raise

    @classmethod
    def IS_CHANGED(s, image, max_width, max_height, force_square, keep_aspect_ratio, 
                  use_half_precision, force_cpu_offload, aggressive_cleanup):
        return float("NaN")
