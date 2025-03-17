import os
import torch
import numpy as np
from PIL import Image, ImageOps
import folder_paths
from server import PromptServer

# Find and import the original LoadImage class to extend it
from comfy.nodes import LoadImage

class ReloadImageNode(LoadImage):
    """
    Custom extension of the LoadImage node with a reload button functionality.
    """
    CATEGORY = "image"
    
    @classmethod
    def INPUT_TYPES(s):
        # Get the original input types from the parent class
        input_types = LoadImage.INPUT_TYPES()
        
        # Add a unique identifier for this specific instance
        # This will be used by the frontend to know which node to reload
        input_types["hidden"] = {
            "node_id": ("STRING", {"default": ""})
        }
        
        return input_types
    
    FUNCTION = "load_image_with_reload"
    RETURN_TYPES = LoadImage.RETURN_TYPES
    RETURN_NAMES = LoadImage.RETURN_NAMES if hasattr(LoadImage, "RETURN_NAMES") else None
    
    def __init__(self):
        super().__init__()
        # Keep track of the last loaded image path
        self.last_image_path = None
    
    def load_image_with_reload(self, image, **kwargs):
        """
        Load an image and remember its path for reloading.
        This function will be called by ComfyUI when the node is executed.
        """
        # Extract node_id from kwargs (passed by ComfyUI)
        node_id = kwargs.pop("node_id", "unknown")
        
        # Store image path for reloading
        image_path = folder_paths.get_full_path("input", image)
        self.last_image_path = image_path
        
        # Register this node instance with the server for reload events
        PromptServer.instance.send_sync("reload_image_node.register", {
            "node_id": node_id,
            "image_name": image
        })
        
        # Use the parent class method to actually load the image
        return super().load_image(image)
    
    @classmethod
    def IS_CHANGED(s, image, **kwargs):
        """
        Check if the image has changed, used by ComfyUI caching system.
        """
        # Always return a unique value to ensure reloading works
        # This is necessary because we want to force a reload when requested
        # but still use the ComfyUI caching system otherwise
        return hash(f"{image}_{os.path.getmtime(folder_paths.get_full_path('input', image))}")
    
# Register a custom API route to handle image reloading
@PromptServer.instance.routes.post("/reload_image")
async def reload_image(request):
    """
    API endpoint to reload an image.
    This will be called by the frontend when the reload button is clicked.
    """
    data = await request.json()
    node_id = data.get("node_id")
    
    # Notify the frontend to trigger a node execution
    PromptServer.instance.send_sync("reload_image_node.reload", {
        "node_id": node_id
    })
    
    return {"success": True}