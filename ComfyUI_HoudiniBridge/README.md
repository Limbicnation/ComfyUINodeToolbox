# ComfyUI Houdini Bridge

A bridge that automatically sends Houdini renders to ComfyUI for processing.

## Quick Setup Guide

### 1. Install the Bridge

1. Copy the entire `ComfyUI_HoudiniBridge` folder to your ComfyUI custom_nodes directory:
   ```bash
   cp -r ComfyUI_HoudiniBridge /path/to/ComfyUI/custom_nodes/
   ```

### 2. Set Up Environment

1. Open your Houdini environment file:
   ```bash
   # Usually located at:
   ~/.houdini/houdini.env
   # or
   /home/username/houdini19.5/houdini.env
   ```

2. Add these lines:
   ```bash
   # Point to where you installed the bridge
   COMFYUI_BRIDGE_PATH = "/path/to/ComfyUI/custom_nodes/ComfyUI_HoudiniBridge"
   
   # This will be set later after saving your ComfyUI workflow
   COMFYUI_WORKFLOW_PATH = "/path/to/your/workflow.json"
   ```

### 3. Set Up ComfyUI Workflow

1. Start ComfyUI
2. Create a workflow:
   - Add HoudiniBridge node
   - Set watch_directory to your Houdini render output path
   - Set file_pattern to "*.jpg" (or your preferred format)
   - Enable auto_reload and wait_for_render
   - Connect to your processing nodes
3. Save the workflow as JSON (right-click -> Save workflow)
4. Update COMFYUI_WORKFLOW_PATH in houdini.env to point to this saved JSON file

### 4. Configure Houdini ROP Node

1. In your OpenGL ROP node:
   - Go to Scripts tab
   - In Post-Render Script field, paste:
     ```python
     import os
     script_path = os.path.expandvars("$COMFYUI_BRIDGE_PATH/houdini_scripts/post_render.py")
     exec(open(script_path).read())
     ```
   - In Output tab, set your render path (e.g., `/media/gero/Qsync_Ubuntu/Qsync/55_Houdini_Projects_Linux/1_3D/Houdini/1_Scenes/StableHoudini_Linux/Render/Temp/render.jpg`)

### 5. Test the Setup

1. Make sure ComfyUI is running
2. In Houdini, click Render in your ROP node
3. The process should:
   - Render your image
   - Automatically queue the ComfyUI workflow
   - Process the render through ComfyUI

## Troubleshooting

If the workflow doesn't start automatically:

1. Check ComfyUI console for errors
2. Verify environment variables:
   ```python
   # In Houdini Python Shell:
   import os
   print(os.getenv("COMFYUI_BRIDGE_PATH"))
   print(os.getenv("COMFYUI_WORKFLOW_PATH"))
   ```
3. Make sure the watch_directory in HoudiniBridge node matches your render output directory
4. Check that ComfyUI is running and accessible at http://127.0.0.1:8188

## Requirements

- ComfyUI running on localhost:8188
- Python packages: watchdog, requests, PIL
- Houdini 19.0 or later

## Directory Structure

```
ComfyUI_HoudiniBridge/
├── __init__.py
├── houdini_bridge.py
├── houdini_scripts/
│   ├── post_render.py
│   └── setup_rop.py
└── README.md
```

## Common Issues

1. "Workflow not found": Make sure COMFYUI_WORKFLOW_PATH points to your saved workflow JSON
2. "ComfyUI not responding": Ensure ComfyUI is running before rendering
3. "File not detected": Verify the watch_directory path matches your render output path
