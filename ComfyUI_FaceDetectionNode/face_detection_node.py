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
                "classifier_type": (["haar", "lbp"], {"default": "haar"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Cropped Faces",)
    FUNCTION = "detect_and_crop_faces"
    CATEGORY = "image/processing"

    def __init__(self):
        try:
            self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.lbp_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'lbpcascade_frontalface_improved.xml')
            
            # Verify cascades loaded successfully
            if self.haar_cascade.empty():
                logger.error("Failed to load Haar cascade classifier")
                self.haar_cascade = None
            if self.lbp_cascade.empty():
                logger.error("Failed to load LBP cascade classifier")
                self.lbp_cascade = None
                
        except Exception as e:
            logger.error(f"Error initializing cascade classifiers: {str(e)}")
            self.haar_cascade = None
            self.lbp_cascade = None

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

    def detect_and_crop_faces(self, image, detection_threshold, min_face_size, padding, output_mode, classifier_type):
        DEBUG = True
        
        if isinstance(image, torch.Tensor):
            if DEBUG:
                logger.info(f"Raw tensor info - Shape: {image.shape}, Type: {image.dtype}, Device: {image.device}")
            
            # Handle high-dimensional inputs by reshaping
            if len(image.shape) > 4:
                logger.warning(f"Input tensor has unusual shape: {image.shape}, attempting to reshape")
                try:
                    *batch_dims, C, H, W = image.shape
                    batch_size = np.prod(batch_dims)
                    image = image.reshape(batch_size, C, H, W)
                except Exception as e:
                    logger.error(f"Failed to reshape tensor: {str(e)}")
                    raise ValueError(f"Cannot process tensor of shape {image.shape}")

            # Ensure we're working with a batch of images
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            
            if len(image.shape) != 4:
                raise ValueError(f"Expected 3D or 4D tensor, got shape: {image.shape}")
            
            B, C, H, W = image.shape
            
            # Convert high-dimensional channels to 3 channels using average pooling
            if C > 4:
                logger.warning(f"Input has {C} channels, converting to RGB")
                image = image[0].view(3, C//3, H, W).mean(dim=1).unsqueeze(0)
                C = 3
            
            # Convert to numpy, ensuring correct format
            try:
                image_np = image[0].permute(1, 2, 0).cpu().numpy()
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)
            except Exception as e:
                logger.error(f"Tensor conversion failed: {str(e)}")
                raise ValueError(f"Failed to convert tensor to numpy array: {str(e)}")
            
            image = image_np

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
        
        # Select appropriate cascade based on classifier_type
        if classifier_type == "lbp":
            if self.lbp_cascade is None:
                logger.warning("LBP cascade not available, falling back to Haar")
                if self.haar_cascade is None:
                    logger.error("No cascade classifiers available")
                    return (torch.zeros((1, 3, 512, 512)),)
                face_cascade = self.haar_cascade
            else:
                face_cascade = self.lbp_cascade
        else:
            if self.haar_cascade is None:
                logger.error("Haar cascade not available")
                return (torch.zeros((1, 3, 512, 512)),)
            face_cascade = self.haar_cascade
        
        try:
            faces = face_cascade.detectMultiScale(
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
        
        # Convert back to tensor with proper dimensions
        result = torch.from_numpy(result).float() / 255.0
        result = result.permute(2, 0, 1).unsqueeze(0)
        
        # Validate output tensor
        assert result.shape[1] == 3, f"Output must have 3 channels, got {result.shape[1]}"
        
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
