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
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Cropped Faces",)
    FUNCTION = "detect_and_crop_faces"
    CATEGORY = "image/processing"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_and_crop_faces(self, image):
        try:
            logger.info(f"Input image type: {type(image)}, shape: {image.shape}, dtype: {image.dtype}")

            # Convert input tensor to numpy array
            if isinstance(image, torch.Tensor):
                image_np = (image.cpu().numpy() * 255).astype(np.uint8)
            else:
                image_np = (image * 255).astype(np.uint8)

            # Handle batch dimension
            if len(image_np.shape) == 4:
                image_np = image_np[0]  # Take first image from batch

            # Ensure correct channel order (H, W, C)
            if image_np.shape[0] == 3:  # If in (C, H, W) format
                image_np = np.transpose(image_np, (1, 2, 0))

            logger.info(f"Processed input shape: {image_np.shape}, dtype: {image_np.dtype}")

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                # Return original image if no faces detected
                return (torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0,)

            # Crop faces from the image
            cropped_faces = []
            for (x, y, w, h) in faces:
                face = image_rgb[y:y+h, x:x+w]
                cropped_faces.append(face)

            # Stack all cropped faces vertically
            if cropped_faces:
                # Resize all faces to the same size (use size of first face)
                target_size = cropped_faces[0].shape[:2]
                resized_faces = []
                for face in cropped_faces:
                    resized = cv2.resize(face, (target_size[1], target_size[0]))
                    resized_faces.append(resized)
                
                # Stack faces vertically
                stacked_faces = np.vstack(resized_faces)
                
                # Convert to torch tensor format (C, H, W)
                stacked_faces = stacked_faces.transpose(2, 0, 1)
                stacked_faces = torch.from_numpy(stacked_faces).float() / 255.0
                
                # Add batch dimension if needed
                if len(stacked_faces.shape) == 3:
                    stacked_faces = stacked_faces.unsqueeze(0)
                
                return (stacked_faces,)
            else:
                # Return original image if no faces were successfully cropped
                return (torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0,)

        except Exception as e:
            logger.error(f"Error in detect_and_crop_faces: {e}")
            raise

    @classmethod
    def IS_CHANGED(s, image):
        return float("NaN")

NODE_CLASS_MAPPINGS = {
    "FaceDetectionNode": FaceDetectionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetectionNode": "Face Detection"
}