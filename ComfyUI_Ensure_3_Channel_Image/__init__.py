from .ensure_3_channel_image_node import Ensure3ChannelImageNode

NODE_CLASS_MAPPINGS = {
    "Ensure3ChannelImageNode": Ensure3ChannelImageNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ensure3ChannelImageNode": "Ensure 3 Channel Image Node"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
