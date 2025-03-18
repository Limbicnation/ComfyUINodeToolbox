"""
Load Image (Reloadable) - A custom node for ComfyUI
Provides image loading functionality with a reload button
"""
import os
import torch
import numpy as np
from PIL import Image, ImageOps
import folder_paths
from server import PromptServer

# Define the node class
class ReloadImageNode:
    """
    Custom node that loads an image with a reload button functionality
    """
    CATEGORY = "image"
    
    @classmethod
    def INPUT_TYPES(s):
        # Get list of available images
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        
        # Define inputs - this matches standard LoadImage structure
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True})
            },
            "hidden": {
                "unique_id": ("UNIQUE", {"default": "0"})  # This should be hidden
            }
        }
    
    FUNCTION = "load_image"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    OUTPUT_NODE = True  # Mark as output node for display
    
    def load_image(self, image, unique_id=None):
        """Load an image and send node info to frontend for reload functionality"""
        print(f"Loading image {image} with unique_id {unique_id}")
        
        # Get the full path to the image
        image_path = folder_paths.get_full_path("input", image)
        if image_path is None:
            raise FileNotFoundError(f"Image '{image}' not found in input directory")
        
        # Load and process the image
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        
        # Handle different image modes
        if img.mode == "RGBA":
            mask = torch.tensor(np.array(img.getchannel('A')).astype(np.float32) / 255.0)
            img = img.convert("RGB")
        elif img.mode == "LA":
            mask = torch.tensor(np.array(img.getchannel('A')).astype(np.float32) / 255.0)
            img = img.convert("RGB")
        else:
            mask = torch.zeros((img.height, img.width), dtype=torch.float32)
            img = img.convert("RGB")
        
        # Convert to tensor format
        img = torch.tensor(np.array(img).astype(np.float32) / 255.0)
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        return (img, mask)
    
    @classmethod
    def IS_CHANGED(s, image, **kwargs):
        """Check if the image has changed for caching purposes"""
        image_path = folder_paths.get_full_path("input", image)
        if not image_path:
            return hash(image)
        return hash(f"{image}_{os.path.getmtime(image_path)}")

# Register reload endpoint
@PromptServer.instance.routes.post("/reload_image")
async def reload_image(request):
    """API endpoint to handle reload requests from the frontend"""
    try:
        data = await request.json()
        node_id = data.get("node_id")
        if node_id:
            print(f"Reload request received for node: {node_id}")
            
            # Tell the frontend to execute this node
            PromptServer.instance.send_sync("comfy.execute_node", {
                "node_id": node_id
            })
            
            return {"success": True}
        else:
            print("Invalid reload request: missing node_id")
            return {"success": False, "error": "Missing node_id"}
    except Exception as e:
        print(f"Error in reload endpoint: {e}")
        return {"success": False, "error": str(e)}

# Node registration
NODE_CLASS_MAPPINGS = {
    "ReloadImageNode": ReloadImageNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReloadImageNode": "Load Image (Reloadable)"
}

# Define path to web directory
WEB_DIRECTORY = "./web/js"