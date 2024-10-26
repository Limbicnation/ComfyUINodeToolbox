import hou
import os

def setup_post_render_script():
    """
    Set up the post-render script in the current ROP node
    """
    try:
        # Get the current node
        node = hou.pwd()
        if not node:
            print("Error: Cannot access current node")
            return
            
        # Get the bridge path from environment
        bridge_path = os.getenv("COMFYUI_BRIDGE_PATH")
        if not bridge_path:
            print("Error: COMFYUI_BRIDGE_PATH environment variable not set")
            return
            
        # Construct the post-render script
        post_render_script = f"""import os
script_path = os.path.join(os.getenv("COMFYUI_BRIDGE_PATH"), "houdini_scripts", "post_render.py")
exec(open(script_path).read())"""
        
        # Set the post-render script
        post_render_parm = node.parm("postrender")
        if post_render_parm:
            post_render_parm.set(post_render_script)
            print("Successfully set post-render script")
            print("Make sure ComfyUI is running before rendering")
        else:
            print("Error: Could not find post-render parameter")
            
    except Exception as e:
        print(f"Error setting up post-render script: {str(e)}")

if __name__ == "__main__":
    setup_post_render_script()
