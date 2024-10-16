import cv2
import numpy as np
import torch
import logging

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
            logger.info(f"Input image shape: {image.shape}")

            # Check if the input is in the unusual (1, 1, 1024) format
            if image.shape == (1, 1, 1024):
                logger.info("Detected unusual input format (1, 1, 1024)")
                # Reshape to a more standard format (32x32 image with 1 channel)
                image = image.view(1, 32, 32)
                logger.info(f"Reshaped image to: {image.shape}")

            # Convert the input tensor to a numpy array
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            
            logger.info(f"Numpy array shape: {image.shape}")

            # Ensure the image is in the correct format (H, W, C) and uint8
            if len(image.shape) == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            elif len(image.shape) == 2:
                # If it's a 2D array, expand to 3D
                image = np.expand_dims(image, axis=2)
                image = np.repeat(image, 3, axis=2)  # Repeat to create 3 channels
            
            image = (image * 255).astype(np.uint8)
            logger.info(f"Processed image shape: {image.shape}")

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(image_rgb)

            cropped_faces = []
            for (x, y, w, h) in faces:
                face = image_rgb[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (1024, 1024), interpolation=cv2.INTER_AREA)
                face_normalized = (face_resized / 255.0).astype(np.float32)
                face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1)
                cropped_faces.append(face_tensor)

            if not cropped_faces:
                logger.warning("No face detected in the image.")
                return (torch.zeros(1, 3, 1024, 1024),)
            
            result = torch.stack(cropped_faces)
            logger.info(f"Output shape: {result.shape}")

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            return (torch.zeros(1, 3, 1024, 1024),)

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return (result,)

    @classmethod
    def IS_CHANGED(s, image):
        return float("NaN")

if __name__ == "__main__":
    dummy_image = torch.rand(3, 1000, 1000)
    face_detector = FaceDetectionNode()
    result = face_detector.detect_and_crop_faces(dummy_image)
    print(f"Number of faces detected: {result[0].shape[0]}")
    print(f"Output shape of each face: {result[0].shape[1:]}")
