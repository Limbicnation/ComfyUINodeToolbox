import cv2
import numpy as np
import torch
import logging
from PIL import Image
from typing import Tuple, List, Optional

logging.basicConfig(level=logging.INFO)
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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Cropped Faces",)
    FUNCTION = "detect_and_crop_faces"
    CATEGORY = "image/processing"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

    def detect_and_crop_faces(self, image, detection_threshold, min_face_size, padding, output_mode):
        # Debug mode for detailed tensor info
        DEBUG = True
        
        # Convert from tensor format if needed
        if isinstance(image, torch.Tensor):
            if DEBUG:
                logger.info(f"Raw tensor info - Shape: {image.shape}, Type: {image.dtype}, Device: {image.device}")
            
            # Handle different tensor formats
            if len(image.shape) == 4:  # BCHW format
                B, C, H, W = image.shape
                if DEBUG:
                    logger.info(f"Detected BCHW format: {B}x{C}x{H}x{W}")
                
                if H < W and C > 4:  # Likely wrong dimension order
                    # Try to detect correct format
                    if DEBUG:
                        logger.info("Attempting to correct dimension order")
                    if W in [1, 3, 4]:  # Width might be channels
                        image = image.permute(0, 3, 1, 2)
                        if DEBUG:
                            logger.info(f"Permuted shape: {image.shape}")
                
                # Extract first image from batch
                image = image[0]
            else:
                raise ValueError(f"Expected 4D tensor [B,C,H,W], got shape: {image.shape}")
            
            # Ensure channels are in valid range
            if image.shape[0] not in [1, 3, 4]:
                # Try one last permute if channels are wrong
                if image.shape[-1] in [1, 3, 4]:
                    image = image.permute(2, 0, 1)
                else:
                    raise ValueError(f"Cannot determine correct channel dimension from shape: {image.shape}")
            
            # Convert to numpy with safety checks
            try:
                image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                if DEBUG:
                    logger.info(f"Final numpy shape: {image.shape}")
            except Exception as e:
                logger.error(f"Tensor conversion failed: {str(e)}")
                raise ValueError(f"Failed to convert tensor to numpy array: {str(e)}")

        # Validate numpy array
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(image)}")

        # Ensure image has correct dimensions and channels
        if len(image.shape) != 3:
            raise ValueError(f"Expected 3D array (H,W,C), got shape: {image.shape}")

        # Convert to RGB if needed
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] != 3:
            raise ValueError(f"Unexpected number of channels: {image.shape[2]}")

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        try:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_face_size, min_face_size)
            )
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return (torch.zeros((1, 3, 512, 512)),)

        if len(faces) == 0:
            logger.warning("No faces detected in image")
            # Return empty image with correct dimensions
            return (torch.zeros((1, 3, 512, 512)),)

        cropped_faces = []
        for x, y, w, h in faces:
            face_img, _ = self.add_padding(image, (x, y, w, h), padding)
            cropped_faces.append(face_img)

        if output_mode == "largest_face":
            largest_face = max(cropped_faces, key=lambda x: x.shape[0] * x.shape[1])
            cropped_faces = [largest_face]

        # Stack faces horizontally
        if len(cropped_faces) > 1:
            max_height = max(face.shape[0] for face in cropped_faces)
            resized_faces = []
            for face in cropped_faces:
                aspect_ratio = face.shape[1] / face.shape[0]
                new_width = int(max_height * aspect_ratio)
                resized = cv2.resize(face, (new_width, max_height))
                resized_faces.append(resized)
            result = np.hstack(resized_faces)
        else:
            result = cropped_faces[0]

        # Convert to tensor format
        result = torch.from_numpy(result).float() / 255.0
        result = result.permute(2, 0, 1).unsqueeze(0)

        return (result,)

    @classmethod
    def IS_CHANGED(self):
        return False

NODE_CLASS_MAPPINGS = {
    "FaceDetectionNode": FaceDetectionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetectionNode": "Face Detection"
}