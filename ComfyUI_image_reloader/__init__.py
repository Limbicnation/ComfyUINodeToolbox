from .image_reloader import DynamicImageLoader

NODE_CLASS_MAPPINGS = {
    "DynamicImageLoader": DynamicImageLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamicImageLoader": "Dynamic Image Loader"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']