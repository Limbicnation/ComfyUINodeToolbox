import cv2
import numpy as np
import torch
import logging
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetectionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Cropped Faces",)
    FUNCTION = "detect_and_crop_faces"
    CATEGORY = "image/processing"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_and_crop_faces(self, image):
        try:
            # Convert input tensor to numpy array
            if isinstance(image, torch.Tensor):
                if len(image.shape) == 4:  # Batch of images
                    image_np = image[0].cpu().numpy()  # Take first image from batch
                else:
                    image_np = image.cpu().numpy()
                
                # Convert from CHW to HWC format if necessary
                if image_np.shape[0] == 3:  # If in CHW format
                    image_np = np.transpose(image_np, (1, 2, 0))
                
                # Scale to 0-255 range
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image

            # Convert to RGB
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"Unexpected image shape: {image_np.shape}")

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces) == 0:
                # If no faces detected, return original image
                # Ensure correct format for ComfyUI (B,C,H,W)
                return (torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0,)

            # Process detected faces
            cropped_faces = []
            for (x, y, w, h) in faces:
                face = image_rgb[y:y+h, x:x+w]
                cropped_faces.append(face)

            if cropped_faces:
                # Resize all faces to the same size (use size of first face)
                target_size = cropped_faces[0].shape[:2]
                resized_faces = []
                
                for face in cropped_faces:
                    resized = cv2.resize(face, (target_size[1], target_size[0]))
                    resized_faces.append(resized)

                # Stack faces vertically
                stacked_faces = np.vstack(resized_faces)
                
                # Convert to correct format for ComfyUI
                # 1. Convert to CHW format
                stacked_faces = stacked_faces.transpose(2, 0, 1)
                # 2. Convert to tensor and add batch dimension
                stacked_faces = torch.from_numpy(stacked_faces).float().unsqueeze(0) / 255.0

                logger.info(f"Output tensor shape: {stacked_faces.shape}")
                return (stacked_faces,)
            else:
                # Return original image if face processing failed
                return (torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0,)

        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            # Return original image in case of error
            return (torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0,)

    @classmethod
    def IS_CHANGED(s, image):
        return float("NaN")

NODE_CLASS_MAPPINGS = {
    "FaceDetectionNode": FaceDetectionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetectionNode": "Face Detection"
}