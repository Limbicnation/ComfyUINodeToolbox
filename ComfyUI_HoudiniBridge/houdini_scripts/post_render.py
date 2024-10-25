import hou
import os
import json
import requests
import time

def trigger_comfyui_workflow(workflow_path, render_path):
    """
    Triggers a ComfyUI workflow after Houdini render completes
    """
    # ComfyUI API endpoint
    api_url = "http://127.0.0.1:8188"
    
    # Load the workflow file
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    # Find the HoudiniBridge node in the workflow
    for node_id, node in workflow.items():
        if node.get("class_type") == "HoudiniBridge":
            # Update the watch directory to match render output
            render_dir = os.path.dirname(render_path)
            node["inputs"]["watch_directory"] = render_dir
            break
    
    # Prepare the prompt
    prompt = {
        "prompt": workflow,
        "client_id": "houdini_bridge"
    }
    
    # Queue the workflow
    response = requests.post(f"{api_url}/prompt", json=prompt)
    if response.status_code == 200:
        print(f"ComfyUI workflow queued successfully")
    else:
        print(f"Error queuing ComfyUI workflow: {response.status_code}")

def main():
    # Get the current node (ROP node)
    node = hou.pwd()
    
    # Get parameters
    workflow_path = node.parm("comfyui_workflow").eval()
    output_path = node.parm("vm_picture").eval()
    
    # Trigger ComfyUI workflow
    trigger_comfyui_workflow(workflow_path, output_path)

if __name__ == "__main__":
    main()
