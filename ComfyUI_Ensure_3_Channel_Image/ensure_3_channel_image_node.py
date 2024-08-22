import torch
import numpy as np
from PIL import Image

class Ensure3ChannelImageNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    FUNCTION = "ensure_3_channels"
    CATEGORY = "Image Processing"

    def ensure_3_channels(self, image):
        # Check if image is a tensor and convert to numpy array if necessary
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # Handle different number of channels
        if image.ndim == 3:  # If the image has no batch dimension (C, H, W)
            image = np.expand_dims(image, axis=0)  # Add batch dimension (1, C, H, W)

        # Convert from (B, C, H, W) to (B, H, W, C) for PIL processing
        image = np.transpose(image, (0, 2, 3, 1))

        if image.shape[-1] == 4:
            # Convert RGBA to RGB
            image = self.convert_rgba_to_rgb(image)
        elif image.shape[-1] == 1:
            # Convert grayscale to RGB
            image = self.convert_grayscale_to_rgb(image)

        # Convert back to tensor and from (B, H, W, C) to (B, C, H, W)
        image = np.transpose(image, (0, 3, 1, 2))
        image = torch.tensor(image, dtype=torch.float32)

        return (image,)

    def convert_rgba_to_rgb(self, images):
        # Convert RGBA images to RGB
        rgb_images = []
        for img in images:
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
            pil_img = pil_img.convert('RGB')
            rgb_images.append(np.array(pil_img).astype(np.float32) / 255.0)
        return np.stack(rgb_images)

    def convert_grayscale_to_rgb(self, images):
        # Convert grayscale images to RGB
        rgb_images = []
        for img in images:
            pil_img = Image.fromarray((img[:, :, 0] * 255).astype(np.uint8))
            pil_img = pil_img.convert('RGB')
            rgb_images.append(np.array(pil_img).astype(np.float32) / 255.0)
        return np.stack(rgb_images)

NODE_CLASS_MAPPINGS = {
    "Ensure3ChannelImageNode": Ensure3ChannelImageNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ensure3ChannelImageNode": "Ensure 3 Channel Image Node"
}
