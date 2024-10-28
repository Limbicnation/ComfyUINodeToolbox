import torch
import gc
import psutil
import os

class MemoryOptimizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),  # Make latent required since it's the main flow
                "max_batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1
                }),
                "max_dimensions": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64
                }),
                "use_half_precision": ("BOOLEAN", {"default": True}),
                "force_cpu_offload": ("BOOLEAN", {"default": False}),
                "aggressive_cleanup": ("BOOLEAN", {"default": True}),
                "unload_models": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE",)  # Changed order to prioritize LATENT
    RETURN_NAMES = ("optimized_latent", "optimized_image")  # Updated names to match new order
    FUNCTION = "optimize_memory"
    CATEGORY = "utils"

    def __init__(self):
        self.loaded_models = set()

    @staticmethod
    def emergency_cleanup():
        """Perform emergency memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    if obj.is_cuda:
                        del obj
            except Exception:
                pass

        gc.collect()

    @staticmethod
    def get_memory_status():
        """Get current memory usage status"""
        memory_status = {
            'cuda_allocated': 0,
            'cuda_cached': 0,
            'ram_used': 0,
            'ram_total': 0
        }

        if torch.cuda.is_available():
            memory_status['cuda_allocated'] = torch.cuda.memory_allocated() / 1024**2
            memory_status['cuda_cached'] = torch.cuda.memory_reserved() / 1024**2

        ram = psutil.virtual_memory()
        memory_status['ram_used'] = ram.used / 1024**2
        memory_status['ram_total'] = ram.total / 1024**2

        return memory_status

    def unload_all_models(self):
        """Unload all loaded models from memory"""
        try:
            for obj in gc.get_objects():
                if torch.is_tensor(obj) or (hasattr(obj, 'parameters') and callable(obj.parameters)):
                    if hasattr(obj, 'cuda') and callable(obj.cuda):
                        obj.cpu()
                    if hasattr(obj, 'to') and callable(obj.to):
                        obj.to('cpu')
                    del obj

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            gc.collect()
            return True
        except Exception as e:
            print(f"Error during model unloading: {str(e)}")
            return False

    def optimize_latent(self, latent, max_batch_size, use_half_precision, force_cpu_offload):
        """Optimize latent tensor"""
        try:
            if latent is None:
                return None

            # Create a copy to avoid modifying the original
            optimized = {
                'samples': latent['samples'].clone(),
            }
            
            # Handle batch size for latents
            if optimized['samples'].shape[0] > max_batch_size:
                optimized['samples'] = optimized['samples'][:max_batch_size]

            # Convert to half precision if requested
            if use_half_precision and optimized['samples'].dtype != torch.float16:
                optimized['samples'] = optimized['samples'].half()

            # Move to CPU if requested
            if force_cpu_offload and optimized['samples'].device.type == 'cuda':
                optimized['samples'] = optimized['samples'].cpu()

            return optimized
        except Exception as e:
            print(f"Error optimizing latent: {str(e)}")
            return latent  # Return original if optimization fails

    def optimize_image(self, image, max_dimensions, use_half_precision, force_cpu_offload):
        """Optimize image tensor"""
        try:
            if image is None:
                return None

            # Create a copy to avoid modifying the original
            optimized = image.clone()
            
            # Get current dimensions
            batch_size, channels, height, width = optimized.shape

            # Resize if needed
            if height > max_dimensions or width > max_dimensions:
                scale = max_dimensions / max(height, width)
                new_height = int(height * scale)
                new_width = int(width * scale)

                try:
                    optimized = torch.nn.functional.interpolate(
                        optimized,
                        size=(new_height, new_width),
                        mode='bilinear',
                        align_corners=False
                    )
                except torch.cuda.OutOfMemoryError:
                    optimized = optimized.cpu()
                    optimized = torch.nn.functional.interpolate(
                        optimized,
                        size=(new_height, new_width),
                        mode='bilinear',
                        align_corners=False
                    )

            # Convert to half precision if requested
            if use_half_precision and optimized.dtype != torch.float16:
                optimized = optimized.half()

            # Move to CPU if requested
            if force_cpu_offload and optimized.device.type == 'cuda':
                optimized = optimized.cpu()

            return optimized
        except Exception as e:
            print(f"Error optimizing image: {str(e)}")
            return image  # Return original if optimization fails

    def optimize_memory(self, latent, max_batch_size, max_dimensions, 
                       use_half_precision, force_cpu_offload, 
                       aggressive_cleanup, unload_models,
                       image=None):
        try:
            # Initial cleanup if requested
            if aggressive_cleanup:
                self.emergency_cleanup()

            # Unload models if requested
            if unload_models:
                self.unload_all_models()

            # Check memory status
            initial_memory = self.get_memory_status()
            if initial_memory['cuda_allocated'] > 0.9 * initial_memory['cuda_cached']:
                force_cpu_offload = True

            # Process latent (required input)
            optimized_latent = self.optimize_latent(
                latent, max_batch_size, use_half_precision, force_cpu_offload
            )

            # Process image if provided (optional input)
            optimized_image = None
            if image is not None:
                optimized_image = self.optimize_image(
                    image, max_dimensions, use_half_precision, force_cpu_offload
                )

            # Final cleanup
            if aggressive_cleanup:
                self.emergency_cleanup()

            return (optimized_latent, optimized_image)

        except torch.cuda.OutOfMemoryError:
            print("Out of Memory error encountered, attempting recovery...")
            self.emergency_cleanup()
            
            if unload_models:
                self.unload_all_models()
            
            if not force_cpu_offload:
                # Retry with CPU offloading
                return self.optimize_memory(
                    latent, max_batch_size, max_dimensions,
                    use_half_precision, True, True, True,
                    image
                )
            else:
                # Return original inputs if optimization fails
                return (latent, image)

    @classmethod
    def IS_CHANGED(s, latent, max_batch_size, max_dimensions, 
                   use_half_precision, force_cpu_offload, 
                   aggressive_cleanup, unload_models,
                   image=None):
        return float("NaN")
