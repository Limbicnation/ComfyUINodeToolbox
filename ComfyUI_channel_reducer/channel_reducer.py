import logging

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
                "target_channels": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "reduce_channels"

    CATEGORY = "Custom/Utilities"

    def reduce_channels(self, image, target_channels):
        input_channels = image.shape[1]
        logger.info(f"Reducing channels from {input_channels} to {target_channels}")

        if input_channels > target_channels:
            # Slicing channels to match the target
            reduced_image = image[:, :target_channels, :, :]
            logger.info("Channel reduction successful")
        elif input_channels < target_channels:
            error_msg = f"Input has fewer channels ({input_channels}) than the required {target_channels}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            reduced_image = image  # No change needed
            logger.info("Input channels match the target channels, no reduction needed.")

        return (reduced_image,)
