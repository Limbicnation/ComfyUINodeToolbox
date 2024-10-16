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
                "image": ("LATENT",),
                "target_channels": ("INT", {"default": 4, "min": 1, "max": 4, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "reduce_channels"

    CATEGORY = "Custom/Utilities"

    def reduce_channels(self, image, target_channels):
        logger.info(f"Input image shape: {image.shape}")

        # Check if the input is in the unusual (1, 1, 1024) format
        if image.shape == (1, 1, 1024):
            # Reshape to (1, 1024, 1, 1)
            image = image.view(1, 1024, 1, 1)
            logger.info("Reshaped unusual input format")

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

        logger.info(f"Output image shape: {output_image.shape}")
        return (output_image,)
