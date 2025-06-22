import cv2
import numpy as np
import torch
import logging
import os
from PIL import Image
from typing import Tuple, List, Optional

# Configure logging level from environment variable
log_level = os.getenv('COMFYUI_FACE_DETECTION_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

class FaceDetectionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "detection_threshold": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1
                }),
                "min_face_size": ("INT", {
                    "default": 64,
                    "min": 32,
                    "max": 512,
                    "step": 8
                }),
                "padding": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 256,
                    "step": 8
                }),
                "output_mode": (["largest_face", "all_faces"],),
            },
            "optional": {
                "classifier_type": (["default", "alternative"], {"default": "default"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Cropped Faces",)
    FUNCTION = "detect_and_crop_faces"
    CATEGORY = "image/processing"

    def __init__(self):
        self.default_cascade = None
        self.alternative_cascade = None
        
        try:
            # Default Haar cascade - most commonly used and well-tested
            default_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(default_path):
                self.default_cascade = cv2.CascadeClassifier(default_path)
                if self.default_cascade.empty():
                    logger.error(f"Failed to load cascade from {default_path}")
                    self.default_cascade = None
            else:
                logger.error(f"Default cascade file not found: {default_path}")
            
            # Alternative Haar cascade - different training, may detect faces missed by default
            alt_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
            if os.path.exists(alt_path):
                self.alternative_cascade = cv2.CascadeClassifier(alt_path)
                if self.alternative_cascade.empty():
                    logger.warning(f"Failed to load alternative cascade from {alt_path}")
                    self.alternative_cascade = None
            else:
                logger.warning(f"Alternative cascade file not found: {alt_path}")
                
        except Exception as e:
            logger.error(f"Error initializing cascade classifiers: {str(e)}")
            self.default_cascade = None
            self.alternative_cascade = None

    def add_padding(self, image: np.ndarray, face_rect: Tuple[int, int, int, int], padding: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Add padding around detected face and handle boundaries"""
        x, y, w, h = face_rect
        height, width = image.shape[:2]
        
        # Calculate padded coordinates
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)
        
        return image[y1:y2, x1:x2], (x1, y1, x2-x1, y2-y1)

    def detect_and_crop_faces(self, image, detection_threshold, min_face_size, padding, output_mode, classifier_type="default"):
        
        # Convert input to numpy array for OpenCV processing
        if isinstance(image, torch.Tensor):
            logger.debug(f"Processing tensor - Shape: {image.shape}, Type: {image.dtype}")
            
            # Ensure 4D tensor [B, H, W, C] and normalize to RGB
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            elif len(image.shape) != 4:
                raise ValueError(f"Expected 3D or 4D tensor, got shape: {image.shape}")
            
            B, H, W, C = image.shape
            
            # Handle different channel configurations
            if C == 1:
                image = image.repeat(1, 1, 1, 3)  # Grayscale to RGB
            elif C == 4:
                image = image[:, :, :, :3]  # RGBA to RGB
            elif C > 4:
                logger.warning(f"Input has {C} channels, using first 3")
                image = image[:, :, :, :3]
            elif C != 3:
                raise ValueError(f"Cannot handle {C} channels")
            
            # Single conversion: tensor -> numpy (uint8)
            image_np = image[0].cpu().numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = np.clip(image_np, 0, 255).astype(np.uint8)
                
        else:
            # Already numpy array
            image_np = image

        # Validate and ensure RGB format
        if not isinstance(image_np, np.ndarray) or len(image_np.shape) != 3:
            raise ValueError(f"Expected 3D numpy array, got {type(image_np)} with shape {getattr(image_np, 'shape', 'unknown')}")
        
        if image_np.shape[2] != 3:
            raise ValueError(f"Expected RGB image (3 channels), got {image_np.shape[2]} channels")

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Select appropriate cascade based on classifier_type
        if classifier_type == "alternative":
            if self.alternative_cascade is None:
                logger.warning("Alternative Haar cascade not available, falling back to default")
                if self.default_cascade is None:
                    logger.error("No cascade classifiers available")
                    return (torch.zeros((1, 512, 512, 3)),)
                face_cascade = self.default_cascade
            else:
                face_cascade = self.alternative_cascade
        else:  # default
            if self.default_cascade is None:
                logger.error("Default Haar cascade not available")
                return (torch.zeros((1, 512, 512, 3)),)
            face_cascade = self.default_cascade
        
        try:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_face_size, min_face_size)
            )
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return (torch.zeros((1, 512, 512, 3)),)

        if len(faces) == 0:
            logger.warning("No faces detected in image")
            # Return empty image with correct dimensions [B, H, W, C]
            return (torch.zeros((1, 512, 512, 3)),)

        cropped_faces = []
        for x, y, w, h in faces:
            face_img, _ = self.add_padding(image_np, (x, y, w, h), padding)
            cropped_faces.append(face_img)

        if output_mode == "largest_face":
            largest_face = max(cropped_faces, key=lambda x: x.shape[0] * x.shape[1])
            cropped_faces = [largest_face]

        # Modified result handling
        if len(cropped_faces) > 1:
            # Resize all faces to same height while maintaining aspect ratio
            max_height = min(512, max(face.shape[0] for face in cropped_faces))
            resized_faces = []
            for face in cropped_faces:
                aspect_ratio = face.shape[1] / face.shape[0]
                new_width = int(max_height * aspect_ratio)
                resized = cv2.resize(face, (new_width, max_height))
                resized_faces.append(resized)
            result = np.hstack(resized_faces)
        else:
            result = cropped_faces[0]
        
        # Ensure result has correct channel count
        if result.shape[2] == 1:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        elif result.shape[2] == 4:
            result = cv2.cvtColor(result, cv2.COLOR_RGBA2RGB)
        
        # Convert back to tensor with proper dimensions [B, H, W, C]
        result = torch.from_numpy(result).float() / 255.0
        result = result.unsqueeze(0)  # Add batch dimension
        
        # Validate output tensor (format: [B, H, W, C])
        assert result.shape[3] == 3, f"Output must have 3 channels, got {result.shape[3]}"
        
        return (result,)

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return False

NODE_CLASS_MAPPINGS = {
    "FaceDetectionNode": FaceDetectionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetectionNode": "Face Detection"
}
