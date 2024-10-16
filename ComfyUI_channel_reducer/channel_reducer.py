import logging
import torch
import numpy as np

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
                "swap_rb": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "reduce_channels"

    CATEGORY = "Custom/Utilities"

    def reduce_channels(self, image, target_channels, swap_rb):
        logger.info(f"Input image shape: {image.shape}, dtype: {image.dtype}")

        # Ensure input is a torch tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        # Handle the unusual (1, 1, 512) shape
        if image.shape == (1, 1, 512):
            logger.info("Detected unusual (1, 1, 512) shape. Reshaping...")
            image = image.view(1, 512, 1, 1)
            image = image.expand(-1, -1, 512, 512)  # Expand to 512x512
            logger.info(f"Reshaped to: {image.shape}")

        # Ensure 4D tensor (B, C, H, W)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        # Move channels to the correct dimension if necessary
        if image.shape[1] != 1 and image.shape[1] != 3 and image.shape[1] != 4:
            image = image.permute(0, 3, 1, 2)

        input_channels = image.shape[1]
        logger.info(f"Input image has {input_channels} channels")
        logger.info(f"Reducing channels to {target_channels}")

        if input_channels > target_channels:
            # Reduce to target channels
            image = image[:, :target_channels, :, :]
            logger.info("Channel reduction successful")
        elif input_channels < target_channels:
            # Pad with zeros to reach target channels
            padding = torch.zeros(image.shape[0], target_channels - input_channels, *image.shape[2:], device=image.device)
            image = torch.cat([image, padding], dim=1)
            logger.info("Channel padding successful")

        # Swap red and blue channels if requested
        if swap_rb and image.shape[1] >= 3:
            logger.info("Swapping red and blue channels")
            red_channel = image[:, 0, :, :].clone()
            image[:, 0, :, :] = image[:, 2, :, :]
            image[:, 2, :, :] = red_channel

        # Ensure the output is in the format expected by ComfyUI (B, H, W, C)
        image = image.permute(0, 2, 3, 1)

        logger.info(f"Output image shape: {image.shape}, dtype: {image.dtype}")
        return (image,)
