import cv2
import numpy as np
import torch
import logging
from PIL import Image  # Import PIL for saving images

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

            # Ensure the image is a numpy array
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            
            if not isinstance(image, np.ndarray):
                raise ValueError(f"Unexpected image type: {type(image)}")

            # Handle different input shapes
            if len(image.shape) == 4:  # (B, C, H, W) or (B, H, W, C)
                image = image[0]  # Take the first image from the batch
            if image.shape[0] == 3:  # (C, H, W)
                image = np.transpose(image, (1, 2, 0))
            elif image.shape[-1] == 1:  # (H, W, 1)
                image = np.squeeze(image, axis=-1)
            elif len(image.shape) != 3 or image.shape[-1] not in [1, 3]:
                raise ValueError(f"Unexpected image shape: {image.shape}")

            logger.info(f"Processed input shape: {image.shape}, dtype: {image.dtype}")

            # Ensure the image is in uint8 format for OpenCV processing
            image = (image * 255).clip(0, 255).astype(np.uint8)

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Crop faces from the image
            cropped_faces = [image_rgb[y:y+h, x:x+w] for (x, y, w, h) in faces]

            return cropped_faces

        except Exception as e:
            logger.error(f"Error in detect_and_crop_faces: {e}")
            raise

    @classmethod
    def IS_CHANGED(s, image):
        return float("NaN")

if __name__ == "__main__":
    dummy_image = torch.rand(1, 3, 1000, 1000)
    face_detector = FaceDetectionNode()
    result = face_detector.detect_and_crop_faces(dummy_image)
    print(f"Output shape: {result[0].shape}, dtype: {result[0].dtype}")
