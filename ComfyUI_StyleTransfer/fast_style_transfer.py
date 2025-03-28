import os
import sys
import numpy as np
import torch
from PIL import Image
import logging

# Configure logging with a cleaner format
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Print module information
print("*"*80)
print("Fast Style Transfer Node - Enhanced Edition")
print(" - Neural style transfer with multi-scale texture synthesis")
print(" - TensorFlow for neural style transfer (when available)")
print(" - Advanced fallback with feature statistics matching")
print("*"*80)

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

print(json.dumps(result))
                """)
                script_path = f.name
            
            # Run the script in a separate process
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=True, text=True, check=False)
            os.unlink(script_path)  # Clean up temp file
            
            if result.returncode == 0:
                try:
                    tf_result = json.loads(result.stdout.strip())
                    USE_TF_HUB = tf_result.get("tf_hub_available", False)
                    if USE_TF_HUB:
                        logging.info(f"TensorFlow Hub available (v{tf_result.get('tf_hub_version', 'unknown')})")
                    elif tf_result.get("tf_available", False):
                        logging.info(f"TensorFlow available (v{tf_result.get('tf_version', 'unknown')}), but TensorFlow Hub not found")
                    else:
                        logging.info("TensorFlow not available, will use fallback style transfer")
                except json.JSONDecodeError:
                    USE_TF_HUB = False
                    logging.warning("Could not detect TensorFlow availability")
            else:
                USE_TF_HUB = False
                logging.warning("TensorFlow detection failed")
                
        except Exception as e:
            USE_TF_HUB = False
            logging.warning(f"Error checking TensorFlow: {e}")

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
        """Apply true style transfer rather than simple blending"""
        # Import os here to ensure it's available
        import os
        
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

            # Force TensorFlow usage for true style transfer when available
            # Check if TensorFlow and TensorFlow Hub are available
            tf_hub_available = False
            try:
                # Set environment variable to fix protobuf compatibility issues
                import os
                os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
                
                import tensorflow as tf
                
                # Patch the TensorFlow compat.v1 module to avoid estimator error
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

            if tf_hub_available:
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
    
    # Load the style transfer model - this is true neural style transfer
    hub_url = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_url)
    
    # Process images
    content_img = load_image(content_path, (output_size, output_size))
    style_img = load_image(style_path, (256, 256))
    
    # Apply neural style transfer - the core of true style transfer
    # This uses a pre-trained neural network to extract and apply style features
    result_tensor = hub_module(tf.constant(content_img), tf.constant(style_img))[0]
    
    # Convert result to numpy array
    result_array = result_tensor.numpy()[0]  # Remove batch dimension
    
    # Apply color preservation if requested
    if preserve_color:
        # Keep color from content image but take style from result
        content_array = tf.image.decode_image(tf.io.read_file(content_path), channels=3).numpy()
        content_array = tf.image.resize(content_array, tf.shape(result_array)[:2], preserve_aspect_ratio=True).numpy()
        content_array = content_array / 255.0  # Normalize to 0-1

        # Convert both to YUV color space
        def rgb_to_yuv(rgb):
            # RGB to YUV conversion
            r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            y = 0.299 * r + 0.587 * g + 0.114 * b
            u = -0.14713 * r - 0.28886 * g + 0.436 * b
            v = 0.615 * r - 0.51499 * g - 0.10001 * b
            return np.stack([y, u, v], axis=-1)
            
        def yuv_to_rgb(yuv):
            # YUV to RGB conversion
            y, u, v = yuv[..., 0], yuv[..., 1], yuv[..., 2]
            r = y + 1.13983 * v
            g = y - 0.39465 * u - 0.58060 * v
            b = y + 2.03211 * u
            return np.stack([r, g, b], axis=-1)
        
        # Convert to YUV
        content_yuv = rgb_to_yuv(content_array)
        style_yuv = rgb_to_yuv(result_array)
        
        # Replace Y channel only to preserve color but transfer style
        combined_yuv = np.copy(content_yuv)
        combined_yuv[..., 0] = style_yuv[..., 0]
        
        # Convert back to RGB
        result_array = yuv_to_rgb(combined_yuv)
        result_array = np.clip(result_array, 0.0, 1.0)  # Ensure valid RGB values
    
    # Convert result to base64 encoded image
    result_img = Image.fromarray((result_array * 255).astype(np.uint8))
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
                    logging.info("Using TensorFlow Hub for neural style transfer (isolated process)")
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
            
            # If we get here, TensorFlow failed or isn't available
            # Use an enhanced fallback that attempts to better simulate neural style transfer
            logging.info("Using enhanced fallback style transfer method")
            return self.enhanced_style_transfer(content_image, style_image, style_weight, preserve_color)
                
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

    def enhanced_style_transfer(self, content_image, style_image, style_weight=1.0, preserve_color=True):
        """Advanced fallback method that simulates neural style transfer"""
        content_np = self.tensor_to_numpy(content_image)
        style_np = self.tensor_to_numpy(style_image)
        
        # Ensure RGB format for both images
        if content_np.ndim == 2:
            content_np = np.stack([content_np, content_np, content_np], axis=2)
        elif content_np.shape[2] == 1:
            content_np = np.repeat(content_np, 3, axis=2)
            
        if style_np.ndim == 2:
            style_np = np.stack([style_np, style_np, style_np], axis=2)
        elif style_np.shape[2] == 1:
            style_np = np.repeat(style_np, 3, axis=2)
        
        # Resize style image to match content image dimensions
        style_img_resized = np.array(Image.fromarray(
            (style_np * 255).astype(np.uint8)).resize(
            (content_np.shape[1], content_np.shape[0])))
        style_img_resized = style_img_resized.astype(np.float32) / 255.0
        
        # Create multi-scale representation for better texture transfer
        # This helps simulate how neural networks capture features at different scales
        def create_pyramid(img, levels=3):
            pyramid = [img]
            for i in range(levels-1):
                # Downsample
                h, w = pyramid[-1].shape[:2]
                down_size = (w//2, h//2)
                down_img = np.array(Image.fromarray(
                    (pyramid[-1] * 255).astype(np.uint8)).resize(down_size))
                pyramid.append(down_img.astype(np.float32) / 255.0)
            return pyramid
        
        # Create image pyramids
        content_pyramid = create_pyramid(content_np)
        style_pyramid = create_pyramid(style_img_resized)
        
        # Process each level and combine
        result = np.zeros_like(content_np)
        
        # Apply style transfer at each level
        for level in range(len(content_pyramid)):
            content_level = content_pyramid[level]
            style_level = style_pyramid[level]
            
            # Extract texture information from style
            # Calculate local statistics to capture texture properties
            def extract_texture_stats(img, window_size=5):
                h, w = img.shape[:2]
                # Initialize arrays for local mean and variance
                local_mean = np.zeros_like(img)
                local_var = np.zeros_like(img)
                
                # Simple convolution to calculate local statistics
                # This is a basic approximation of texture extraction
                padding = window_size // 2
                padded = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
                
                for i in range(h):
                    for j in range(w):
                        window = padded[i:i+window_size, j:j+window_size, :]
                        local_mean[i, j, :] = np.mean(window, axis=(0, 1))
                        local_var[i, j, :] = np.var(window, axis=(0, 1))
                
                return local_mean, local_var
            
            # Extract content and style statistics
            c_mean, c_var = extract_texture_stats(content_level, window_size=5)
            s_mean, s_var = extract_texture_stats(style_level, window_size=5)
            
            # Apply texture transfer
            # This is a simplified version of neural style transfer statistics matching
            c_std = np.sqrt(c_var)
            s_std = np.sqrt(s_var)
            
            # Generate texture-transferred image
            level_result = c_mean + c_std * (style_level - s_mean) / (s_std + 1e-6)
            level_result = np.clip(level_result, 0.0, 1.0)
            
            # If color preservation is enabled, apply in YUV space
            if preserve_color:
                # Convert to YUV
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
                
                # Convert images to YUV
                content_yuv = rgb_to_yuv(content_level)
                result_yuv = rgb_to_yuv(level_result)
                
                # Transfer only Y channel
                combined_yuv = content_yuv.copy()
                combined_yuv[..., 0] = result_yuv[..., 0]
                
                # Convert back to RGB
                level_result = yuv_to_rgb(combined_yuv)
                level_result = np.clip(level_result, 0.0, 1.0)
            
            # Resize result back to original size if needed
            if level > 0:
                level_result = np.array(Image.fromarray(
                    (level_result * 255).astype(np.uint8)).resize(
                    (content_np.shape[1], content_np.shape[0])))
                level_result = level_result.astype(np.float32) / 255.0
            
            # Add to final result with appropriate weighting
            # Higher pyramid levels (smaller images) handle larger features
            # Lower pyramid levels handle details
            weight = 1.0 / (2 ** level)
            result += level_result * weight
        
        # Normalize result
        result = result / np.sum([1.0 / (2 ** l) for l in range(len(content_pyramid))])
        
        # Apply final contrast enhancement to make style more pronounced
        mean_val = np.mean(result, axis=(0, 1), keepdims=True)
        result = mean_val + (result - mean_val) * (1.0 + style_weight * 0.5)
        result = np.clip(result, 0.0, 1.0)
        
        # Convert back to tensor
        return torch.from_numpy(result).unsqueeze(0).permute(0, 3, 1, 2)
        
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
            
            # Apply contrast enhancement to style Y channel to emphasize texture patterns
            style_y = style_yuv[..., 0]
            mean_style = np.mean(style_y)
            style_y_enhanced = mean_style + (style_y - mean_style) * 1.2  # Boost contrast by 20%
            style_yuv[..., 0] = np.clip(style_y_enhanced, 0.0, 1.0)
            
            # Blend the Y (luminance) channel, keep UV (chrominance) from content
            alpha = min(max(0.3, style_weight / 4.0), 0.9)  # Stronger style influence
            result_yuv = content_yuv.copy()
            result_yuv[..., 0] = (1 - alpha) * content_yuv[..., 0] + alpha * style_yuv[..., 0]
            
            # Convert back to RGB
            result = yuv_to_rgb(result_yuv)
        else:
            # Enhanced RGB blending with contrast boost
            style_enhanced = style_img_resized.copy()
            
            # Enhance contrast to emphasize style patterns
            mean_style = np.mean(style_enhanced, axis=(0, 1), keepdims=True)
            style_enhanced = mean_style + (style_enhanced - mean_style) * 1.3  # Boost contrast by 30%
            style_enhanced = np.clip(style_enhanced, 0.0, 1.0)
            
            # Apply stronger alpha for direct style transfer
            alpha = min(max(0.3, style_weight / 4.0), 0.9)
            result = (1 - alpha) * content_np + alpha * style_enhanced
        
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