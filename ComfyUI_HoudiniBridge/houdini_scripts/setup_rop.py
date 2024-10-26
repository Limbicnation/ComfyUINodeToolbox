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
            
        # Print all parameters for debugging
        print("\nAvailable parameters:")
        for parm in node.parms():
            print(f"- {parm.name()}: {parm.description()}")
            
        # Get the bridge path from environment
        bridge_path = os.getenv("COMFYUI_BRIDGE_PATH")
        if not bridge_path:
            print("\nError: COMFYUI_BRIDGE_PATH environment variable not set")
            return
            
        # Construct the post-render script
        post_render_script = f"""import os
script_path = os.path.join(os.getenv("COMFYUI_BRIDGE_PATH"), "houdini_scripts", "post_render.py")
exec(open(script_path).read())"""
        
        # Try different possible parameter names
        possible_params = ['postscript', 'postrender', 'postframe', 'postwritescript']
        found_param = None
        
        for param_name in possible_params:
            param = node.parm(param_name)
            if param:
                found_param = param
                print(f"\nFound parameter: {param_name}")
                break
                
        if found_param:
            found_param.set(post_render_script)
            print("Successfully set post-render script")
            print("Make sure ComfyUI is running before rendering")
        else:
            print("\nError: Could not find post-render script parameter.")
            print("Available script parameters might be:")
            for parm in node.parms():
                if 'script' in parm.name().lower():
                    print(f"- {parm.name()}: {parm.description()}")
            
    except Exception as e:
        print(f"Error setting up post-render script: {str(e)}")

if __name__ == "__main__":
    setup_post_render_script()
