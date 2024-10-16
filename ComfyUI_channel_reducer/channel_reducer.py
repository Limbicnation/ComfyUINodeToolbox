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
                "image": ("IMAGE",),  # Changed from LATENT to IMAGE
                "target_channels": ("INT", {"default": 3, "min": 1, "max": 4, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "reduce_channels"

    CATEGORY = "Custom/Utilities"

    def reduce_channels(self, image, target_channels):
        logger.info(f"Input image shape: {image.shape}, dtype: {image.dtype}")

        # Convert to torch tensor if it's a numpy array
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

        input_channels = image.shape[1]
        logger.info(f"Input image has {input_channels} channels")
        logger.info(f"Reducing channels to {target_channels}")

        if input_channels > 4:
            # First, reduce to 4 channels
            reduced_image = image[:, :4, :, :]
            input_channels = 4
        else:
            reduced_image = image

        if input_channels > target_channels:
            # Further reduce to target channels
            reduced_image = reduced_image[:, :target_channels, :, :]
            logger.info("Channel reduction successful")
        elif input_channels < target_channels:
            # Pad with zeros to reach target channels
            padding = torch.zeros(reduced_image.shape[0], target_channels - input_channels, *reduced_image.shape[2:])
            reduced_image = torch.cat([reduced_image, padding], dim=1)
            logger.info("Channel padding successful")
        else:
            logger.info("Input channels match the target channels, no change needed.")

        # Ensure the output is in a format that can be handled by PIL.Image.fromarray()
        output_image = reduced_image.squeeze().permute(1, 2, 0).cpu().numpy()
        output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)

        logger.info(f"Output image shape: {output_image.shape}, dtype: {output_image.dtype}")
        return (output_image,)
