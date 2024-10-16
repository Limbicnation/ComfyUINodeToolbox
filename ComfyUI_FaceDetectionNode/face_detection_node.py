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

            # Convert the input tensor to a numpy array
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            
            logger.info(f"Numpy array shape: {image.shape}")

            # Ensure the image is in the correct format (H, W, C) and uint8
            if len(image.shape) == 3:
                if image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))
                elif image.shape[2] != 3:
                    image = np.squeeze(image)
                    image = np.stack((image,)*3, axis=-1)
            elif len(image.shape) == 2:
                image = np.stack((image,)*3, axis=-1)
            
            image = (image * 255).clip(0, 255).astype(np.uint8)
            logger.info(f"Processed image shape: {image.shape}")

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(image_rgb)

            cropped_faces = []
            for (x, y, w, h) in faces:
                face = image_rgb[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (1024, 1024), interpolation=cv2.INTER_AREA)
                cropped_faces.append(face_resized)

            if not cropped_faces:
                logger.warning("No face detected in the image.")
                return (np.zeros((1, 1024, 1024, 3), dtype=np.uint8),)
            
            result = np.stack(cropped_faces)
            logger.info(f"Output shape: {result.shape}")

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            return (np.zeros((1, 1024, 1024, 3), dtype=np.uint8),)

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
