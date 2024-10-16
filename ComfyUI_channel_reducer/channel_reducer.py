import logging
import torch
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChannelReducer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_channels": ("INT", {"default": 3, "min": 1, "max": 4, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "reduce_channels"

    CATEGORY = "Custom/Utilities"

    def reduce_channels(self, image, target_channels):
        logger.info(f"Input image shape: {image.shape}, dtype: {image.dtype}")

        # Convert to numpy array if it's a torch tensor
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # Handle the unusual (1, 1, 512) shape
        if image.shape == (1, 1, 512):
            logger.info("Detected unusual (1, 1, 512) shape. Reshaping...")
            image = image.reshape(512, 1, 1)
            image = np.broadcast_to(image, (512, 512, 1))
            logger.info(f"Reshaped to: {image.shape}")

        # Ensure the image is in the correct format (H, W, C)
        if len(image.shape) == 4:
            image = image.squeeze(0)  # Remove batch dimension if present
        if image.shape[0] == 1 or image.shape[0] == 3 or image.shape[0] == 4:
            image = np.transpose(image, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)

        # Convert to uint8 if necessary
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Create PIL Image
        pil_image = Image.fromarray(image)

        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Reduce channels
        if target_channels == 1:
            reduced_image = pil_image.convert('L')
        elif target_channels == 3:
            reduced_image = pil_image.convert('RGB')
        elif target_channels == 4:
            reduced_image = pil_image.convert('RGBA')
        else:
            raise ValueError(f"Unsupported number of target channels: {target_channels}")

        # Convert back to numpy array
        output_image = np.array(reduced_image)

        logger.info(f"Output image shape: {output_image.shape}, dtype: {output_image.dtype}")
        return (output_image,)
