from .nodes import ReloadImageNode

NODE_CLASS_MAPPINGS = {
    "ReloadImageNode": ReloadImageNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReloadImageNode": "Load Image (Reloadable)"
}

WEB_DIRECTORY = "./web/js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']