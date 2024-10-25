import hou

def add_comfyui_parameters():
    """
    Adds ComfyUI-related parameters to the ROP node
    """
    node = hou.pwd()
    
    # Create a new tab for ComfyUI settings
    ptg = node.parmTemplateGroup()
    
    # Create a new folder for ComfyUI parameters
    comfyui_folder = hou.FolderParmTemplate("comfyui", "ComfyUI")
    
    # Add workflow file parameter
    workflow_parm = hou.StringParmTemplate(
        "comfyui_workflow",
        "Workflow File",
        1,
        default_value=[""],
        file_type=hou.fileType.Any,
        tags={"filechooser_pattern": "*.json"}
    )
    
    # Add post-render script parameter
    script_parm = hou.StringParmTemplate(
        "comfyui_post_script",
        "Post-Render Script",
        1,
        default_value=["python %s/houdini_scripts/post_render.py" % "$COMFYUI_BRIDGE_PATH"]
    )
    
    # Add parameters to folder
    comfyui_folder.addParmTemplate(workflow_parm)
    comfyui_folder.addParmTemplate(script_parm)
    
    # Add folder to parameter template group
    ptg.append(comfyui_folder)
    
    # Set the new parameter template group
    node.setParmTemplateGroup(ptg)
    
    # Set up post-render script
    node.parm("postrender").set("`chs(\"comfyui_post_script\")`")

if __name__ == "__main__":
    add_comfyui_parameters()
