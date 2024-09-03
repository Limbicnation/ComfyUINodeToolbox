import os
import certifi
import requests
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import torch
from PIL import Image
import logging

# Set SSL_CERT_FILE environment variable
os.environ['SSL_CERT_FILE'] = certifi.where()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class FastStyleTransferNode:
    CATEGORY = "Image/Style Transfer"

    @classmethod    
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content_image": ("IMAGE",),
                "style_image": ("IMAGE",),
                "output_image_size": ("INT", {"default": 384, "min": 1}),
                "target_height": ("INT", {"default": 1024, "min": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_style_transfer"

    @staticmethod
    def validate_image_size(image_path, min_size=(10, 10)):
        """Validates that the image at image_path is at least min_size."""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
    
        if img.shape[0] < min_size[0] or img.shape[1] < min_size[1]:
            raise ValueError(f"Image at {image_path} is too small: {img.shape}. Minimum size required: {min_size}.")
        return img

    def load_image(self, image_path, image_size=(256, 256), preserve_aspect_ratio=True):
        """Loads an image from a file and resizes it."""
        # Validate image size before processing
        self.validate_image_size(image_path)

        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)

        # Logging: Print original image shape
        logging.debug(f"Original image dimensions: {img.shape}")

        # Check if the image has valid dimensions
        if img.shape[0] <= 0 or img.shape[1] <= 0:
            # Provide a fallback image
            logging.warning("Invalid image dimensions detected. Returning a placeholder image.")
            return torch.zeros((1, 3, image_size[1], image_size[0]))

        img = tf.image.convert_image_dtype(img, tf.float32)

        # Logging: Print target resize dimensions
        logging.debug(f"Target resize dimensions: {image_size}")

        # Ensure the image dimensions are valid before resizing
        if image_size[0] > 0 and image_size[1] > 0:
            img = tf.image.resize(img, image_size, preserve_aspect_ratio=preserve_aspect_ratio)
        else:
            raise ValueError(f"Invalid target size for resizing: {image_size}")

        # Logging: Print resized image shape
        logging.debug(f"Resized image dimensions: {img.shape}")

        if img.shape[0] <= 0 or img.shape[1] <= 0:
            raise ValueError(f"Resized image has invalid dimensions: {img.shape}")

        return img[tf.newaxis, :]

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
        # Ensure the NumPy array has 3 channels (RGB)
        if np_array.ndim == 3 and np_array.shape[-1] == 1:
            np_array = np.repeat(np_array, 3, axis=-1)  # Convert grayscale to RGB by repeating channels
        elif np_array.ndim == 4 and np_array.shape[-1] == 1:
            np_array = np.repeat(np_array, 3, axis=-1)  # Handle batch dimension for grayscale

        # Ensure the array is in the correct range (0-255) and type (uint8)
        img = Image.fromarray((np_array * 255).astype(np.uint8))
        img.save(image_path)

    def apply_style_transfer(self, content_image, style_image, output_image_size=384, target_height=1024):
        # Prepare paths for temporary files
        temp_dir = './temp_images'
        os.makedirs(temp_dir, exist_ok=True)
        content_image_path = os.path.join(temp_dir, 'temp_content_image.jpg')
        style_image_path = os.path.join(temp_dir, 'temp_style_image.jpg')

        try:
            # Convert torch.Tensor to numpy arrays and ensure they have 3 channels
            content_np = self.tensor_to_numpy(content_image)
            style_np = self.tensor_to_numpy(style_image)

            # Save numpy arrays as images
            self.save_numpy_as_image(content_np, content_image_path)
            self.save_numpy_as_image(style_np, style_image_path)

            # Use requests to download the TensorFlow Hub module
            hub_url = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
            response = requests.get(hub_url, verify=certifi.where())
            if response.status_code == 200:
                hub_module = hub.load(hub_url)

            # Load and resize the images using TensorFlow
            content_img = self.load_image(content_image_path, (output_image_size, output_image_size))
            style_img = self.load_image(style_image_path, (256, 256))

            # Logging: Check dimensions and channels
            logging.debug(f"Content image shape before model: {content_img.shape}")
            logging.debug(f"Style image shape before model: {style_img.shape}")

            # Apply the style transfer
            stylized_image = hub_module(tf.constant(content_img), tf.constant(style_img))[0]

            # Logging: Check the output shape and channels
            logging.debug(f"Stylized image shape after model: {stylized_image.shape}")

            # Convert the stylized image back to a PyTorch tensor
            stylized_image = stylized_image.numpy()

            # Ensure the output image has 3 channels
            if stylized_image.shape[-1] == 1:
                stylized_image = np.repeat(stylized_image, 3, axis=-1)

            stylized_image = torch.from_numpy(stylized_image).permute(0, 3, 1, 2)

            return stylized_image
        finally:
            # Clean up temporary files
            if os.path.exists(content_image_path):
                os.remove(content_image_path)
            if os.path.exists(style_image_path):
                os.remove(style_image_path)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)

# __init__.py content
from .fast_style_transfer import FastStyleTransferNode

NODE_CLASS_MAPPINGS = {
    "FastStyleTransferNode": FastStyleTransferNode,
}

__all__ = ['NODE_CLASS_MAPPINGS']
