import cv2
import numpy as np
import torch

class FaceDetectionNode:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Cropped Face",)
    FUNCTION = "detect_and_crop_face"
    CATEGORY = "image/processing"

    def __init__(self):
        # Load the pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    def detect_and_crop_face(self, image):
        # Convert the input tensor to a numpy array
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Ensure the image is in the correct format (H, W, C) and uint8
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        image = (image * 255).astype(np.uint8)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            # Get the first detected face
            x, y, w, h = faces[0]

            # Crop the face
            face = image[y:y+h, x:x+w]

            # Resize to 1024x1024
            face_resized = cv2.resize(face, (1024, 1024), interpolation=cv2.INTER_AREA)

            # Convert back to RGB (OpenCV uses BGR)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            # Normalize to 0-1 range and convert to float32
            face_normalized = face_rgb.astype(np.float32) / 255.0

            # Convert to tensor and change to channel-first format
            face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1)

            return (face_tensor,)
        else:
            print("No face detected in the image.")
            return (torch.zeros(3, 1024, 1024),)  # Return a blank image if no face is detected

# This part is for testing the node independently
if __name__ == "__main__":
    # Create a dummy image tensor
    dummy_image = torch.rand(3, 256, 256)

    # Instantiate the node
    face_detector = FaceDetectionNode()

    # Call the node's function
    result = face_detector.detect_and_crop_face(dummy_image)

    print(f"Output shape: {result[0].shape}")
