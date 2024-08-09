import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import torch
from PIL import Image
from server import PromptServer  # Assuming PromptServer is available in ComfyUI

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
        img = tf.image.decode_image(img, channels=3)
    
        if img.shape[0] < min_size[0] or img.shape[1] < min_size[1]:
            raise ValueError(f"Image at {image_path} is too small: {img.shape}. Minimum size required: {min_size}.")
        return img

    def load_image(self, image_path, image_size=(256, 256), preserve_aspect_ratio=True):
        """Loads an image from a file and resizes it."""
        # Validate image size before processing
        self.validate_image_size(image_path)

        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)

        # Debugging: Print original image shape
        print(f"Original image dimensions: {img.shape}")

        # Check if the image has valid dimensions
        if img.shape[0] <= 0 or img.shape[1] <= 0:
            # Provide a fallback image
            print("Invalid image dimensions detected. Returning a placeholder image.")
            return torch.zeros((1, 3, image_size[1], image_size[0]))

        img = tf.image.convert_image_dtype(img, tf.float32)

        # Debugging: Print target resize dimensions
        print(f"Target resize dimensions: {image_size}")

        # Ensure the image dimensions are valid before resizing
        if image_size[0] > 0 and image_size[1] > 0:
            img = tf.image.resize(img, image_size, preserve_aspect_ratio=preserve_aspect_ratio)
        else:
            raise ValueError(f"Invalid target size for resizing: {image_size}")

        # Debugging: Print resized image shape
        print(f"Resized image dimensions: {img.shape}")

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
        # Handle dimensions (assume [Height, Width, Channels])
        if np_array.ndim == 3:
            img = Image.fromarray((np_array * 255).astype(np.uint8))
        elif np_array.ndim == 4:
            # Handle the case with a batch dimension by squeezing it
            img = Image.fromarray((np_array.squeeze(0) * 255).astype(np.uint8))
        else:
            raise ValueError(f"Unexpected array shape: {np_array.shape}")

        img.save(image_path)

    def apply_style_transfer(self, content_image, style_image, output_image_size=384, target_height=1024):
        # Prepare paths for temporary files
        temp_dir = './temp_images'
        os.makedirs(temp_dir, exist_ok=True)
        content_image_path = os.path.join(temp_dir, 'temp_content_image.jpg')
        style_image_path = os.path.join(temp_dir, 'temp_style_image.jpg')

        # Convert torch.Tensor to numpy arrays and ensure they have 3 channels
        content_np = self.tensor_to_numpy(content_image)
        style_np = self.tensor_to_numpy(style_image)

        # Save numpy arrays as images
        self.save_numpy_as_image(content_np, content_image_path)
        self.save_numpy_as_image(style_np, style_image_path)

        # Load the TensorFlow Hub model
        hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

        # Load and resize the images using TensorFlow
        content_img = self.load_image(content_image_path, (output_image_size, output_image_size))
        style_img = self.load_image(style_image_path, (256, 256))

        # Add debug information to check dimensions before applying the model
        print(f"Content image shape: {content_img.shape}")
        print(f"Style image shape: {style_img.shape}")

        # Check that the images have valid dimensions
        if content_img.shape[1] <= 0 or content_img.shape[2] <= 0:
            raise ValueError(f"Invalid content image dimensions after resize: {content_img.shape}")
        if style_img.shape[1] <= 0 or style_img.shape[2] <= 0:
            raise ValueError(f"Invalid style image dimensions after resize: {style_img.shape}")

        # Apply the style transfer
        stylized_image = hub_module(tf.constant(content_img), tf.constant(style_img))[0]

        # Convert the stylized image back to a PyTorch tensor
        stylized_image = stylized_image.numpy()
        stylized_image = torch.from_numpy(stylized_image).permute(0, 3, 1, 2)

        return stylized_image
