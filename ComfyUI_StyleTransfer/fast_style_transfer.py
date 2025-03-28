import os
import sys
import numpy as np
import torch
from PIL import Image
import logging

# Configure logging with a less verbose level
logging.basicConfig(level=logging.INFO)

# Safely try to import optional dependencies
try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
except ImportError:
    logging.info("certifi not available, skipping SSL certificate setup")

class FastStyleTransferNode:
    CATEGORY = "Image/Style Transfer"

    @classmethod    
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content_image": ("IMAGE",),
                "style_image": ("IMAGE",),
                "output_image_size": ("INT", {"default": 384, "min": 1}),
                "style_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_style_transfer"

    def validate_image_size(self, image_path, min_size=(10, 10)):
        """Validates that the image at image_path is at least min_size."""
        # Check if TensorFlow is available
        tf_available = False
        try:
            # Set environment variable to fix protobuf compatibility issues
            import os
            os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
            
            import tensorflow as tf
            
            # Patch the TensorFlow compat.v1 module to avoid estimator error if needed
            if not hasattr(tf.compat.v1, 'estimator'):
                class DummyEstimator:
                    class Exporter:
                        pass
                # Add a dummy estimator to avoid the AttributeError
                setattr(tf.compat.v1, 'estimator', DummyEstimator)
                
            tf_available = True
        except ImportError:
            pass
            
        if tf_available:
            # TensorFlow-based validation
            img = tf.io.read_file(image_path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            
            if img.shape[0] < min_size[0] or img.shape[1] < min_size[1]:
                raise ValueError(f"Image at {image_path} is too small: {img.shape}. Minimum size required: {min_size}.")
            return img
        else:
            # PIL-based validation
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    if width < min_size[1] or height < min_size[0]:
                        raise ValueError(f"Image at {image_path} is too small: {(height, width)}. Minimum size required: {min_size}.")
                    return np.array(img)
            except Exception as e:
                logging.error(f"Error validating image size: {e}")
                return None

    def load_image(self, image_path, image_size=(256, 256), preserve_aspect_ratio=True):
        """Loads an image from a file and resizes it."""
        # Check if TensorFlow is available
        tf_available = False
        try:
            # Set environment variable to fix protobuf compatibility issues
            import os
            os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
            
            import tensorflow as tf
            
            # Patch the TensorFlow compat.v1 module to avoid estimator error if needed
            if not hasattr(tf.compat.v1, 'estimator'):
                class DummyEstimator:
                    class Exporter:
                        pass
                # Add a dummy estimator to avoid the AttributeError
                setattr(tf.compat.v1, 'estimator', DummyEstimator)
                
            tf_available = True
        except ImportError:
            pass
            
        if tf_available:
            # TensorFlow-based image loading
            # Validate image size before processing
            self.validate_image_size(image_path)

            img = tf.io.read_file(image_path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            logging.debug(f"Original image dimensions: {img.shape}")

            # Check if the image has valid dimensions
            if img.shape[0] <= 0 or img.shape[1] <= 0:
                logging.warning("Invalid image dimensions detected. Returning a placeholder image.")
                return tf.zeros([1, image_size[0], image_size[1], 3], dtype=tf.float32)

            img = tf.image.convert_image_dtype(img, tf.float32)
            logging.debug(f"Target resize dimensions: {image_size}")

            # Ensure the image dimensions are valid before resizing
            if image_size[0] > 0 and image_size[1] > 0:
                img = tf.image.resize(img, image_size, preserve_aspect_ratio=preserve_aspect_ratio)
            else:
                raise ValueError(f"Invalid target size for resizing: {image_size}")

            logging.debug(f"Resized image dimensions: {img.shape}")
            return img[tf.newaxis, :]
        else:
            # PIL-based image loading
            try:
                with Image.open(image_path) as img:
                    # Validate size
                    width, height = img.size
                    if width <= 0 or height <= 0:
                        logging.warning("Invalid image dimensions detected. Returning a placeholder image.")
                        return np.zeros([1, image_size[0], image_size[1], 3], dtype=np.float32)
                    
                    # Resize image
                    if preserve_aspect_ratio:
                        # Calculate new dimensions preserving aspect ratio
                        ratio = min(image_size[0] / height, image_size[1] / width)
                        new_height = int(height * ratio)
                        new_width = int(width * ratio)
                        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                    else:
                        resized_img = img.resize(image_size[::-1], Image.LANCZOS)  # PIL uses (width, height)
                    
                    # Convert to numpy array and normalize
                    np_img = np.array(resized_img, dtype=np.float32) / 255.0
                    
                    # Ensure 3 channels
                    if np_img.ndim == 2:  # Grayscale
                        np_img = np.stack([np_img, np_img, np_img], axis=-1)
                    elif np_img.shape[-1] == 4:  # RGBA
                        np_img = np_img[:, :, :3]
                    
                    # Add batch dimension
                    return np_img[np.newaxis, :]
            except Exception as e:
                logging.error(f"Error loading image: {e}")
                return np.zeros([1, image_size[0], image_size[1], 3], dtype=np.float32)

    def tensor_to_numpy(self, tensor):
        """Converts a PyTorch tensor to a NumPy array."""
        np_array = tensor.cpu().numpy()

        # Handle cases where the tensor has a batch dimension
        if np_array.ndim == 4:
            np_array = np_array.squeeze(0)  # Remove batch dimension if present

        # Ensure the array is in the correct format (float32 or uint8)
        if np_array.dtype != np.float32:
            np_array = np_array.astype(np.float32)
        
        return np_array

    def save_numpy_as_image(self, np_array, image_path):
        """Saves a NumPy array as an image file."""
        # Import os here to ensure it's available
        import os
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # Ensure the NumPy array has 3 channels (RGB)
        if np_array.ndim == 3 and np_array.shape[-1] == 1:
            np_array = np.repeat(np_array, 3, axis=-1)  # Convert grayscale to RGB by repeating channels
        elif np_array.ndim == 4 and np_array.shape[-1] == 1:
            np_array = np.repeat(np_array, 3, axis=-1)  # Handle batch dimension for grayscale

        # Ensure the array is in the correct range (0-255) and type (uint8)
        img = Image.fromarray((np_array * 255).astype(np.uint8))
        img.save(image_path)

    def apply_style_transfer(self, content_image, style_image, output_image_size=384, style_weight=1.0):
        # Import os here to ensure it's available
        import os
        
        # Prepare paths for temporary files
        temp_dir = './temp_images'
        os.makedirs(temp_dir, exist_ok=True)
        content_image_path = os.path.join(temp_dir, 'temp_content_image.jpg')
        style_image_path = os.path.join(temp_dir, 'temp_style_image.jpg')

        # Check if TensorFlow and TensorFlow Hub are available
        tf_hub_available = False
        try:
            # Set environment variable to fix protobuf compatibility issues
            import os
            os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
            
            import tensorflow as tf
            
            # Patch the TensorFlow compat.v1 module to avoid estimator error
            # This is a workaround for the missing estimator module in some TensorFlow installations
            if not hasattr(tf.compat.v1, 'estimator'):
                logging.info("Patching TensorFlow to handle missing estimator module")
                class DummyEstimator:
                    class Exporter:
                        pass
                # Add a dummy estimator to avoid the AttributeError
                setattr(tf.compat.v1, 'estimator', DummyEstimator)
            
            try:
                # Set PYTHONPATH to avoid issues with TensorFlow Hub importing
                import sys
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                
                # Set protobuf implementation to pure Python to avoid descriptor errors
                logging.info("Using pure-Python implementation of protobuf for TensorFlow Hub compatibility")
                
                # Import TensorFlow Hub with patched environment
                import tensorflow_hub as hub
                tf_hub_available = True
                logging.info("TensorFlow Hub available, will use enhanced style transfer")
            except ImportError:
                logging.info("TensorFlow Hub not available, will use fallback style transfer method")
            except AttributeError as e:
                logging.info(f"TensorFlow Hub attribute error: {e}. Using fallback.")
        except ImportError:
            logging.info("TensorFlow not available, will use fallback image processing")

        try:
            # Convert torch.Tensor to numpy arrays and ensure they have 3 channels
            content_np = self.tensor_to_numpy(content_image)
            style_np = self.tensor_to_numpy(style_image)

            # Save numpy arrays as images
            self.save_numpy_as_image(content_np, content_image_path)
            self.save_numpy_as_image(style_np, style_image_path)

            if tf_hub_available:
                logging.info("Using TensorFlow Hub for style transfer")
                # Use TensorFlow Hub module
                hub_url = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
                try:
                    # Create a monkeypatch for the estimator module if needed
                    def monkeypatch_hub_load():
                        # Store the original __import__ function
                        original_import = __import__
                        
                        # Define our custom import function
                        def custom_import(name, globals=None, locals=None, fromlist=(), level=0):
                            # If trying to import tensorflow_estimator, return a dummy module
                            if name == 'tensorflow_estimator' or name.startswith('tensorflow_estimator.'):
                                class DummyEstimatorModule:
                                    class python:
                                        class estimator:
                                            class SessionRunHook:
                                                pass
                                            class EvalSpec:
                                                pass
                                            class TrainSpec:
                                                pass
                                            class Exporter:
                                                pass
                                            class LatestExporter:
                                                pass
                                return DummyEstimatorModule()
                            # For everything else, use the original import
                            return original_import(name, globals, locals, fromlist, level)
                        
                        # Replace the built-in __import__ function
                        __builtins__['__import__'] = custom_import
                        
                        try:
                            # Try loading the hub module with our patched import
                            module = hub.load(hub_url)
                            return module
                        finally:
                            # Restore the original __import__ function
                            __builtins__['__import__'] = original_import
                    
                    # Try to load the module with our monkeypatch
                    try:
                        hub_module = monkeypatch_hub_load()
                    except Exception as import_error:
                        logging.error(f"Advanced TensorFlow Hub monkeypatching failed: {import_error}")
                        # Try the regular way as fallback
                        hub_module = hub.load(hub_url)
                    
                    # Load and resize the images using TensorFlow
                    content_img = self.load_image(content_image_path, (output_image_size, output_image_size))
                    style_img = self.load_image(style_image_path, (256, 256))
                    
                    # Apply the style transfer
                    stylized_image = hub_module(tf.constant(content_img), tf.constant(style_img))[0]
                    
                    # Convert the stylized image back to a numpy array
                    stylized_image = stylized_image.numpy()
                except Exception as e:
                    logging.error(f"TensorFlow Hub style transfer failed: {e}")
                    # Fall back to alternative method
                    stylized_image = self.fallback_style_transfer(content_np, style_np, style_weight)
            else:
                logging.info("Using fallback style transfer method")
                stylized_image = self.fallback_style_transfer(content_np, style_np, style_weight)

            # Ensure the output image has 3 channels
            if stylized_image.ndim == 3 and stylized_image.shape[-1] == 1:
                stylized_image = np.repeat(stylized_image, 3, axis[-1])
            
            # Convert to PyTorch tensor format
            if tf_hub_available:
                stylized_tensor = torch.from_numpy(stylized_image).permute(0, 3, 1, 2)
            else:
                stylized_tensor = torch.from_numpy(stylized_image).unsqueeze(0).permute(0, 3, 1, 2)
                
            return stylized_tensor
        finally:
            # Import os here to make sure it's available in the finally block
            import os
            
            # Clean up temporary files
            if os.path.exists(content_image_path):
                os.remove(content_image_path)
            if os.path.exists(style_image_path):
                os.remove(style_image_path)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
                
    def fallback_style_transfer(self, content_img, style_img, style_weight=1.0):
        """Simple fallback method for style transfer when TensorFlow Hub is not available"""
        logging.info("Using basic style blending as fallback")
        
        # Resize style image to match content image dimensions
        style_img_resized = np.array(Image.fromarray(
            (style_img * 255).astype(np.uint8)).resize(
            (content_img.shape[1], content_img.shape[0])))
        style_img_resized = style_img_resized.astype(np.float32) / 255.0
        
        # Simple blending of content and style
        # This is a very basic approach - just a weighted average
        alpha = min(max(0.2, style_weight / 5.0), 0.8)  # Convert style_weight to alpha in range [0.2, 0.8]
        result = (1 - alpha) * content_img + alpha * style_img_resized
        
        # Ensure result is in proper range
        result = np.clip(result, 0.0, 1.0)
        
        return result

# __init__.py content
from .fast_style_transfer import FastStyleTransferNode

NODE_CLASS_MAPPINGS = {
    "FastStyleTransferNode": FastStyleTransferNode,
}

__all__ = ['NODE_CLASS_MAPPINGS']
