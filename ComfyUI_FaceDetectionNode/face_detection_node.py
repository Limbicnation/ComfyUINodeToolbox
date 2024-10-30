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
            # Log input shape
            logger.info(f"Input shape: {image.shape}, type: {type(image)}")

            # Convert input tensor to numpy array
            if isinstance(image, torch.Tensor):
                if len(image.shape) == 4:  # Batch of images
                    image_np = image[0].cpu().numpy()  # Take first image from batch
                else:
                    image_np = image.cpu().numpy()
            else:
                image_np = image

            # Log numpy array shape after conversion
            logger.info(f"Numpy array shape: {image_np.shape}")

            # Convert from CHW to HWC if necessary
            if image_np.shape[0] == 3:  # If in CHW format
                image_np = np.transpose(image_np, (1, 2, 0))
                logger.info(f"After transpose: {image_np.shape}")

            # Ensure we're working with float values 0-1
            if image_np.dtype != np.float32:
                image_np = image_np.astype(np.float32)
            
            # Scale to 0-255 range for OpenCV
            image_np_uint8 = (image_np * 255).clip(0, 255).astype(np.uint8)
            
            # Convert to RGB for OpenCV
            image_rgb = cv2.cvtColor(image_np_uint8, cv2.COLOR_BGR2RGB)
            
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
                logger.info("No faces detected")
                # Return original image in correct format
                output = torch.from_numpy(image_np_uint8).float() / 255.0
                if len(output.shape) == 3:
                    output = output.permute(2, 0, 1)  # HWC to CHW
                    output = output.unsqueeze(0)  # Add batch dimension
                return (output,)

            # Process detected faces
            cropped_faces = []
            max_h, max_w = 0, 0
            
            # First pass: determine maximum dimensions
            for (x, y, w, h) in faces:
                max_h = max(max_h, h)
                max_w = max(max_w, w)

            # Second pass: crop and resize faces
            for (x, y, w, h) in faces:
                face = image_rgb[y:y+h, x:x+w]
                # Resize to maximum dimensions to ensure consistent size
                face_resized = cv2.resize(face, (max_w, max_h))
                cropped_faces.append(face_resized)

            if cropped_faces:
                # Stack faces vertically
                stacked_faces = np.vstack(cropped_faces)
                
                # Convert to float32 and normalize to 0-1
                stacked_faces = stacked_faces.astype(np.float32) / 255.0
                
                # Convert to torch tensor
                output = torch.from_numpy(stacked_faces)
                
                # Ensure correct channel order (HWC to CHW)
                if len(output.shape) == 3:
                    output = output.permute(2, 0, 1)
                
                # Add batch dimension if needed
                if len(output.shape) == 3:
                    output = output.unsqueeze(0)

                logger.info(f"Final output shape: {output.shape}")
                return (output,)
            else:
                logger.warning("Failed to process faces")
                # Return original image as fallback
                output = torch.from_numpy(image_np_uint8).float() / 255.0
                if len(output.shape) == 3:
                    output = output.permute(2, 0, 1)
                    output = output.unsqueeze(0)
                return (output,)

        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            # Return original image in case of error
            output = torch.from_numpy(image_np_uint8).float() / 255.0
            if len(output.shape) == 3:
                output = output.permute(2, 0, 1)
                output = output.unsqueeze(0)
            return (output,)

    @classmethod
    def IS_CHANGED(s, image):
        return float("NaN")

NODE_CLASS_MAPPINGS = {
    "FaceDetectionNode": FaceDetectionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetectionNode": "Face Detection"
}