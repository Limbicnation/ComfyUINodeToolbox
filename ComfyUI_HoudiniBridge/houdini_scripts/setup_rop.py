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
            
        # Print node type and all parameters
        print(f"\nNode type: {node.type().name()}")
        print("\nALL Available parameters:")
        for parm in node.parms():
            print(f"- {parm.name()}: {parm.description()}")
            
        # Print template group info
        ptg = node.parmTemplateGroup()
        print("\nParameter folders:")
        for folder in ptg.entries():
            print(f"- {folder.label()}")
            
        # Get the bridge path from environment
        bridge_path = os.getenv("COMFYUI_BRIDGE_PATH")
        if not bridge_path:
            print("\nError: COMFYUI_BRIDGE_PATH environment variable not set")
            return
            
        # Construct the post-render script
        post_render_script = f"""import os
script_path = os.path.join(os.getenv("COMFYUI_BRIDGE_PATH"), "houdini_scripts", "post_render.py")
exec(open(script_path).read())"""

        print("\nLooking for script parameters...")
        # Try to find any parameter that might be for scripts
        for parm in node.parms():
            parm_name = parm.name().lower()
            parm_desc = parm.description().lower()
            if ('script' in parm_name or 'script' in parm_desc or 
                'python' in parm_name or 'python' in parm_desc):
                print(f"Potential script parameter found: {parm.name()} - {parm.description()}")
            
    except Exception as e:
        print(f"Error in setup script: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    setup_post_render_script()
