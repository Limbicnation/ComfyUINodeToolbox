# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

Since this is a ComfyUI node collection without a traditional build system, most development is done directly with ComfyUI:

- **Testing nodes**: Load ComfyUI with the nodes and test in the UI
- **Dependencies**: Install via pip as needed (see StyleTransfer requirements.txt for optional deps)
- **Main dependencies**: `pip install tensorflow==2.11.0 tensorflow-hub==0.12.0 keras==2.11.0` (for StyleTransfer features)

## Repository Architecture

This is a collection of 13 independent ComfyUI custom nodes, each implementing different image processing and utility functions. Each node is self-contained in its own directory following the pattern:

```
ComfyUI_{NodeName}/
├── __init__.py                  # Node registration (NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)
└── {implementation}.py          # Main node class with INPUT_TYPES, RETURN_TYPES, FUNCTION
```

### Key Architectural Patterns

**Node Registration**: All nodes use the standard ComfyUI registration pattern via `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in `__init__.py`.

**Class Structure**: Each node class implements:
- `INPUT_TYPES()` classmethod defining required/optional inputs
- `RETURN_TYPES` tuple defining output types  
- `FUNCTION` string naming the main execution method
- `CATEGORY` for UI organization

**Tensor Conventions**: All nodes follow ComfyUI's tensor format `[Batch, Height, Width, Channels]` with float32 values in 0-1 range.

### Node Categories

**Image Processing**: StyleTransfer (neural style transfer), FaceDetectionNode (OpenCV face detection), channel manipulation nodes
**Utilities**: MemoryOptimizer (GPU/CPU management), RandomSeedGenerator (cross-library seed sync), ImageReloader (dynamic loading)
**Integration**: HoudiniBridge (external 3D software integration)
**Specialized**: NaiveBayesNode (statistical computations)

### Version Compatibility

The FaceDetectionNode demonstrates advanced ComfyUI version compatibility - it detects ComfyUI v3 API availability and provides dual implementations, using modern schema definitions when available while falling back to legacy patterns for v1/v2.

### External Dependencies

Most nodes use core dependencies (torch, numpy, PIL). Optional dependencies include:
- opencv-cv2 (face detection)
- tensorflow + tensorflow-hub (style transfer)  
- scikit-image (enhanced color processing)
- psutil, watchdog, requests (utility functions)

## Node Development Guidelines

When creating new nodes:
1. Follow the established directory structure and naming conventions
2. Implement proper error handling and fallback strategies
3. Ensure tensor format compatibility (BHWC, float32, 0-1 range)
4. Add proper logging with configurable levels via environment variables
5. Consider memory management for GPU/CPU operations
6. Implement `IS_CHANGED()` method for caching behavior

## Integration Patterns

The collection demonstrates several integration approaches:
- **Pipeline Integration**: Standard ComfyUI workflow nodes
- **External Tool Integration**: Houdini bridge with HTTP API communication
- **Web UI Extensions**: Custom JavaScript for enhanced interactions (see reload_image_node)
- **Cross-Platform**: Graceful handling of CUDA availability and library dependencies