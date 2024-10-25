import hou
import os
import json
import requests
import time

def get_workflow_path():
    """
    Get the workflow path from environment variable or default location
    """
    workflow_path = os.getenv("COMFYUI_WORKFLOW_PATH")
    if not workflow_path:
        print("Error: COMFYUI_WORKFLOW_PATH environment variable not set")
        print("Please set it to your saved workflow JSON file path")
        return None
    return os.path.expandvars(os.path.expanduser(workflow_path))

def trigger_comfyui_workflow(node):
    """
    Triggers a ComfyUI workflow after Houdini render completes
    """
    try:
        # ComfyUI API endpoint
        api_url = "http://127.0.0.1:8188"
        
        # Get the render output directory
        render_path = node.parm('picture').eval()
        render_dir = os.path.dirname(render_path)
        
        # Wait for render to complete
        time.sleep(1)
        
        # Get workflow file path
        workflow_path = get_workflow_path()
        if not workflow_path or not os.path.exists(workflow_path):
            print(f"Error: ComfyUI workflow file not found at {workflow_path}")
            return
            
        # Load the workflow file
        try:
            with open(workflow_path, 'r') as f:
                workflow = json.load(f)
        except Exception as e:
            print(f"Error loading workflow file: {str(e)}")
            return
            
        # Find the HoudiniBridge node in the workflow
        bridge_found = False
        for node_id, node_data in workflow.items():
            if node_data.get("class_type") == "HoudiniBridge":
                # Update the watch directory
                node_data["inputs"]["watch_directory"] = render_dir
                # Set file pattern based on render output extension
                ext = os.path.splitext(render_path)[1]
                node_data["inputs"]["file_pattern"] = f"*{ext}"
                bridge_found = True
                break
                
        if not bridge_found:
            print("Error: No HoudiniBridge node found in workflow")
            return
            
        # Prepare the prompt
        prompt = {
            "prompt": workflow,
            "client_id": "houdini_bridge"
        }
        
        # Queue the workflow
        try:
            response = requests.post(f"{api_url}/prompt", json=prompt)
            if response.status_code == 200:
                print(f"ComfyUI workflow queued successfully")
                print(f"Watching directory: {render_dir}")
            else:
                print(f"Error queuing ComfyUI workflow: {response.status_code}")
                print(f"Response: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to ComfyUI: {str(e)}")
            print("Make sure ComfyUI is running and accessible at http://127.0.0.1:8188")
            
    except Exception as e:
        print(f"Unexpected error in post-render script: {str(e)}")

def main():
    """
    Main entry point for post-render script
    """
    try:
        # Get the current node (ROP node)
        node = hou.pwd()
        if not node:
            print("Error: Cannot access current node")
            return
            
        # Trigger the workflow
        trigger_comfyui_workflow(node)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
