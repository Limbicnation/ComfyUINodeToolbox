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
            
            logger.info(f"Processed input shape: {image.shape}, dtype: {image.dtype}")

            # Ensure the image is in uint8 format for OpenCV processing
            image = (image * 255).clip(0, 255).astype(np.uint8)

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            logger.info(f"Number of faces detected: {len(faces)}")

            if len(faces) == 0:
                logger.warning("No face detected in the image. Returning the original image.")
                result = image_rgb
            else:
                x, y, w, h = faces[0]  # Take the first face
                face = image_rgb[y:y+h, x:x+w]
                result = cv2.resize(face, (512, 512), interpolation=cv2.INTER_AREA)

            # Convert result to float32 and normalize to [0, 1]
            result = result.astype(np.float32) / 255.0

            # Convert numpy array to torch tensor
            result = torch.from_numpy(result).float()
            
            # Ensure the tensor is in the format (B, C, H, W)
            if len(result.shape) == 3:
                result = result.permute(2, 0, 1).unsqueeze(0)
            elif len(result.shape) == 2:
                result = result.unsqueeze(0).unsqueeze(0)
                result = result.repeat(1, 3, 1, 1)

            # Ensure dtype is float32
            result = result.to(torch.float32)

            logger.info(f"Final output shape: {result.shape}, dtype: {result.dtype}")

            # Convert the result back to numpy for saving with PIL
            output_image = result.squeeze().permute(1, 2, 0).cpu().numpy()
            output_image = (output_image * 255).clip(0, 255).astype(np.uint8)

            # Save the result using PIL
            output_pil_image = Image.fromarray(output_image)
            output_pil_image.save("output_face.png")

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            # Return original image if an error occurs
            result = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
            result = result.permute(2, 0, 1).unsqueeze(0)
            result = result.to(torch.float32)

        return (result,)

    @classmethod
    def IS_CHANGED(s, image):
        return float("NaN")

if __name__ == "__main__":
    dummy_image = torch.rand(1, 3, 1000, 1000)
    face_detector = FaceDetectionNode()
    result = face_detector.detect_and_crop_faces(dummy_image)
    print(f"Output shape: {result[0].shape}, dtype: {result[0].dtype}")
