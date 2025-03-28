import os
import sys
import numpy as np
import torch
from PIL import Image
import logging

# Configure logging with a cleaner format
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Initialize flag for TensorFlow availability
USE_TF_HUB = False

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
                "preserve_color": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_style_transfer"
    
    def __init__(self):
        # Try to safely import TensorFlow and TensorFlow Hub only once during initialization
        global USE_TF_HUB
        
        try:
            # Use a separate process to check TensorFlow availability without affecting main environment
            import subprocess
            import json
            import tempfile
            
            # Create a temporary Python script to check TensorFlow
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
                f.write("""
import os
import json
import sys

# Set environment variables for TensorFlow
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

result = {"tf_available": False, "tf_hub_available": False, "error": None}

try:
    import tensorflow as tf
    result["tf_available"] = True
    result["tf_version"] = tf.__version__
    
    # Add dummy estimator if needed
    if not hasattr(tf.compat.v1, 'estimator'):
        class DummyEstimator:
            class Exporter: pass
        setattr(tf.compat.v1, 'estimator', DummyEstimator)
    
    try:
        import tensorflow_hub as hub
        result["tf_hub_available"] = True
        result["tf_hub_version"] = hub.__version__
    except Exception as e:
        result["error"] = str(e)
except Exception as e:
    result["error"] = str(e)

# Print JSON result for parent process to capture
print(json.dumps(result))
                """)
                script_path = f.name
            
            # Run the script in a separate process
            result = subprocess.run([sys.executable, script_path], 
                                    capture_output=True, text=True, check=False)
            os.unlink(script_path)  # Clean up temp file
            
            if result.returncode == 0:
                try:
                    tf_check = json.loads(result.stdout.strip())
                    if tf_check.get("tf_hub_available", False):
                        USE_TF_HUB = True
                        logging.info(f"TensorFlow {tf_check.get('tf_version')} and TensorFlow Hub {tf_check.get('tf_hub_version')} available")
                    elif tf_check.get("tf_available", False):
                        logging.info(f"TensorFlow {tf_check.get('tf_version')} available, but TensorFlow Hub could not be loaded")
                        if tf_check.get("error"):
                            logging.info(f"TensorFlow Hub error: {tf_check.get('error')}")
                    else:
                        logging.info("TensorFlow not available, will use fallback method")
                        if tf_check.get("error"):
                            logging.info(f"TensorFlow error: {tf_check.get('error')}")
                except json.JSONDecodeError:
                    logging.warning("Could not parse TensorFlow availability check results")
            else:
                logging.warning(f"TensorFlow check failed: {result.stderr}")
                
        except Exception as e:
            logging.warning(f"Could not check TensorFlow availability: {e}")
            logging.info("Will use fallback style transfer method")

    def tensor_to_numpy(self, tensor):
        """Converts a PyTorch tensor to a NumPy array."""
        np_array = tensor.cpu().numpy()

        # Handle cases where the tensor has a batch dimension
        if np_array.ndim == 4:
            np_array = np_array.squeeze(0)  # Remove batch dimension if present

        # Ensure the array is in the correct format (float32 or uint8)
        if np_array.dtype != np.float32:
            np_array = np_array.astype(np.float32)
            
        # Ensure we have a 3D array with 3 channels (RGB)
        if np_array.ndim == 2:  # Single channel without dimension
            np_array = np.stack([np_array, np_array, np_array], axis=2)
        elif np_array.ndim == 3 and np_array.shape[2] == 1:  # Single channel with dimension
            np_array = np.repeat(np_array, 3, axis=2)
        
        return np_array

    def save_numpy_as_image(self, np_array, image_path):
        """Saves a NumPy array as an image file."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(image_path) or '.', exist_ok=True)
        
        # Ensure the NumPy array has 3 channels (RGB)
        if np_array.ndim == 2:  # Single channel without dimension
            np_array = np.stack([np_array, np_array, np_array], axis=2)
        elif np_array.ndim == 3 and np_array.shape[2] == 1:  # Single channel with dimension
            np_array = np.repeat(np_array, 3, axis=2)

        # Ensure the array is in the correct range (0-255) and type (uint8)
        img = Image.fromarray((np_array * 255).astype(np.uint8))
        
        # Explicitly convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img.save(image_path)

    def apply_style_transfer(self, content_image, style_image, output_image_size=384, style_weight=1.0, preserve_color=True):
        # Prepare paths for temporary files
        temp_dir = './temp_images'
        os.makedirs(temp_dir, exist_ok=True)
        content_image_path = os.path.join(temp_dir, 'temp_content_image.jpg')
        style_image_path = os.path.join(temp_dir, 'temp_style_image.jpg')

        try:
            # Convert torch.Tensor to numpy arrays
            content_np = self.tensor_to_numpy(content_image)
            style_np = self.tensor_to_numpy(style_image)

            # Save numpy arrays as images
            self.save_numpy_as_image(content_np, content_image_path)
            self.save_numpy_as_image(style_np, style_image_path)

            if USE_TF_HUB:
                try:
                    # Try to use TensorFlow in a separate process to avoid environment conflicts
                    import subprocess
                    import json
                    import tempfile
                    import base64
                    
                    # Create a temporary Python script for TensorFlow processing
                    with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
                        f.write(f"""
import os
import sys
import json
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# Set environment variables for TensorFlow
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Parse inputs
content_path = "{content_image_path}"
style_path = "{style_image_path}"
output_size = {output_image_size}
preserve_color = {str(preserve_color).lower()}

result = {{"success": False, "error": None}}

try:
    import tensorflow as tf
    
    # Add dummy estimator if needed
    if not hasattr(tf.compat.v1, 'estimator'):
        class DummyEstimator:
            class Exporter: pass
        setattr(tf.compat.v1, 'estimator', DummyEstimator)
    
    import tensorflow_hub as hub
    
    # Load and process images
    def load_image(image_path, size):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, size, preserve_aspect_ratio=True)
        return img[tf.newaxis, :]
    
    # Load the style transfer model
    hub_url = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_url)
    
    # Process images
    content_img = load_image(content_path, (output_size, output_size))
    style_img = load_image(style_path, (256, 256))
    
    # Apply style transfer
    result_tensor = hub_module(tf.constant(content_img), tf.constant(style_img))[0]
    result_array = result_tensor.numpy()[0]  # Remove batch dimension
    
    # Apply color preservation if requested
    if preserve_color:
        # Keep color from content image but take style from result
        content_array = tf.image.decode_image(tf.io.read_file(content_path), channels=3).numpy()
        content_array = tf.image.resize(content_array, tf.shape(result_array)[:2], preserve_aspect_ratio=True).numpy()
        content_array = content_array / 255.0  # Normalize to 0-1

        # Convert both to YUV color space
        def rgb_to_yuv(rgb):
            # Simple RGB to YUV conversion
            r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            y = 0.299 * r + 0.587 * g + 0.114 * b
            u = -0.14713 * r - 0.28886 * g + 0.436 * b
            v = 0.615 * r - 0.51499 * g - 0.10001 * b
            return np.stack([y, u, v], axis=-1)
            
        def yuv_to_rgb(yuv):
            # Simple YUV to RGB conversion
            y, u, v = yuv[..., 0], yuv[..., 1], yuv[..., 2]
            r = y + 1.13983 * v
            g = y - 0.39465 * u - 0.58060 * v
            b = y + 2.03211 * u
            return np.stack([r, g, b], axis=-1)
        
        # Convert to YUV
        content_yuv = rgb_to_yuv(content_array)
        style_yuv = rgb_to_yuv(result_array)
        
        # Replace the Y channel of the content with Y from style
        combined_yuv = np.copy(content_yuv)
        combined_yuv[..., 0] = style_yuv[..., 0]
        
        # Convert back to RGB
        result_array = yuv_to_rgb(combined_yuv)
        result_array = np.clip(result_array, 0.0, 1.0)  # Ensure valid RGB values
    
    # Ensure the result is a 3-channel RGB image
    if result_array.ndim == 2:
        result_array = np.stack([result_array, result_array, result_array], axis=-1)
    elif result_array.ndim == 3 and result_array.shape[-1] == 1:
        result_array = np.repeat(result_array, 3, axis=-1)
    
    # Convert result to base64 encoded image
    result_img = Image.fromarray((result_array * 255).astype(np.uint8))
    if result_img.mode != 'RGB':
        result_img = result_img.convert('RGB')
    
    buffer = BytesIO()
    result_img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    result = {{"success": True, "image": img_str}}
except Exception as e:
    result["error"] = str(e)

# Print JSON result for parent process to capture
print(json.dumps(result))
                        """)
                        script_path = f.name
                    
                    # Run the script in a separate process
                    logging.info("Using TensorFlow Hub for style transfer (isolated process)")
                    result = subprocess.run([sys.executable, script_path], 
                                          capture_output=True, text=True, check=False)
                    os.unlink(script_path)  # Clean up temp file
                    
                    if result.returncode == 0:
                        try:
                            tf_result = json.loads(result.stdout.strip())
                            if tf_result.get("success", False):
                                # Decode the base64 image
                                import base64
                                from io import BytesIO
                                
                                img_data = base64.b64decode(tf_result["image"])
                                img = Image.open(BytesIO(img_data))
                                
                                # Ensure the image is in RGB mode
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                    
                                stylized_image = np.array(img).astype(np.float32) / 255.0
                                
                                # Convert to PyTorch tensor format
                                stylized_tensor = torch.from_numpy(stylized_image).unsqueeze(0).permute(0, 3, 1, 2)
                                return stylized_tensor
                            else:
                                if tf_result.get("error"):
                                    logging.error(f"TensorFlow Hub error: {tf_result.get('error')}")
                        except json.JSONDecodeError:
                            logging.warning("Could not parse TensorFlow processing results")
                    else:
                        logging.warning(f"TensorFlow processing failed: {result.stderr}")
                        
                except Exception as e:
                    logging.error(f"Error using isolated TensorFlow: {e}")
            
            # Fallback to color-preserving style blending
            logging.info("Using fallback style transfer method")
            return self.fallback_style_transfer(content_image, style_image, style_weight, preserve_color)
                
        finally:
            # Clean up temporary files
            if os.path.exists(content_image_path):
                os.remove(content_image_path)
            if os.path.exists(style_image_path):
                os.remove(style_image_path)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
                
    def fallback_style_transfer(self, content_image, style_image, style_weight=1.0, preserve_color=True):
        """Enhanced fallback method for style transfer with color preservation"""
        content_np = self.tensor_to_numpy(content_image)
        style_np = self.tensor_to_numpy(style_image)
        
        # Resize style image to match content image dimensions
        style_img_resized = np.array(Image.fromarray(
            (style_np * 255).astype(np.uint8)).resize(
            (content_np.shape[1], content_np.shape[0])))
        style_img_resized = style_img_resized.astype(np.float32) / 255.0
        
        if preserve_color:
            # Convert RGB to YUV for better color preservation
            def rgb_to_yuv(rgb):
                r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
                y = 0.299 * r + 0.587 * g + 0.114 * b
                u = -0.14713 * r - 0.28886 * g + 0.436 * b
                v = 0.615 * r - 0.51499 * g - 0.10001 * b
                return np.stack([y, u, v], axis=-1)
                
            def yuv_to_rgb(yuv):
                y, u, v = yuv[..., 0], yuv[..., 1], yuv[..., 2]
                r = y + 1.13983 * v
                g = y - 0.39465 * u - 0.58060 * v
                b = y + 2.03211 * u
                return np.stack([r, g, b], axis=-1)
            
            # Convert to YUV color space
            content_yuv = rgb_to_yuv(content_np)
            style_yuv = rgb_to_yuv(style_img_resized)
            
            # Blend the Y (luminance) channel, keep UV (chrominance) from content
            alpha = min(max(0.2, style_weight / 5.0), 0.8)  # Map weight to alpha range [0.2, 0.8]
            result_yuv = content_yuv.copy()
            result_yuv[..., 0] = (1 - alpha) * content_yuv[..., 0] + alpha * style_yuv[..., 0]
            
            # Convert back to RGB
            result = yuv_to_rgb(result_yuv)
        else:
            # Simple RGB blending (won't preserve colors)
            alpha = min(max(0.2, style_weight / 5.0), 0.8)
            result = (1 - alpha) * content_np + alpha * style_img_resized
        
        # Ensure result is in proper range
        result = np.clip(result, 0.0, 1.0)
        
        # Convert back to tensor
        return torch.from_numpy(result).unsqueeze(0).permute(0, 3, 1, 2)

# For __init__.py
NODE_CLASS_MAPPINGS = {
    "FastStyleTransferNode": FastStyleTransferNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FastStyleTransferNode": "Fast Style Transfer"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']