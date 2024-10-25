import hou

def add_comfyui_parameters():
    """
    Adds ComfyUI-related parameters to the ROP node
    """
    try:
        node = hou.pwd()
        if not node:
            print("Error: Cannot access current node")
            return
            
        # Create a new tab for ComfyUI settings
        ptg = node.parmTemplateGroup()
        
        # Check if ComfyUI tab already exists
        if ptg.findFolder("ComfyUI"):
            print("ComfyUI parameters already exist")
            return
            
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
        
        # Add parameter to folder
        comfyui_folder.addParmTemplate(workflow_parm)
        
        # Add folder to parameter template group
        ptg.append(comfyui_folder)
        
        # Set the new parameter template group
        node.setParmTemplateGroup(ptg)
        
        print("Successfully added ComfyUI parameters")
        print("1. Set your workflow JSON file in the ComfyUI tab")
        print("2. Make sure your output path is set correctly")
        print("3. The post-render script will handle the ComfyUI triggering")
        
    except Exception as e:
        print(f"Error setting up ComfyUI parameters: {str(e)}")

if __name__ == "__main__":
    add_comfyui_parameters()
