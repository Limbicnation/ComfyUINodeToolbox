# ComfyUI Houdini Bridge

A bridge node for ComfyUI that connects with Houdini's render output, allowing seamless integration between Houdini renders and ComfyUI workflows.

## Features

- Watches for new renders from Houdini
- Automatically loads rendered images into ComfyUI
- Supports PNG, JPG, and EXR formats
- Configurable watch directory and file patterns
- Optional auto-reload and render wait functionality
- Houdini ROP integration for automatic workflow triggering

## Setup

1. Install the node in your ComfyUI custom_nodes directory
2. Set the COMFYUI_BRIDGE_PATH environment variable in Houdini:
   ```
   setenv COMFYUI_BRIDGE_PATH "/path/to/ComfyUI_HoudiniBridge"
   ```
3. In Houdini, run the setup script on your ROP node:
   ```python
   node = hou.pwd()
   exec(open("$COMFYUI_BRIDGE_PATH/houdini_scripts/setup_rop.py").read())
   ```

## Houdini Integration

1. Select your ROP node
2. Run the setup script to add ComfyUI parameters
3. Set the workflow file path in the ComfyUI tab
4. The post-render script will automatically trigger ComfyUI processing

## Parameters

### ComfyUI Node
- watch_directory: Directory to monitor for new renders
- file_pattern: File pattern to match (e.g., *.png, *.exr)
- auto_reload: Automatically reload when new renders appear
- wait_for_render: Wait for new renders before proceeding

### Houdini ROP
- Workflow File: Path to your ComfyUI workflow JSON file
- Post-Render Script: Script to trigger ComfyUI (auto-configured)

## Workflow

1. Set up your ComfyUI workflow with the HoudiniBridge node
2. Save the workflow as a JSON file
3. Configure the ROP node with the workflow file path
4. Render in Houdini - ComfyUI will automatically process the render

## Requirements

- watchdog
- OpenEXR (for EXR support)
- PIL
- numpy
- torch
- requests (for API communication)
