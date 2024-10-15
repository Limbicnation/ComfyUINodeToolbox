import cv2
import numpy as np
import torch

class FaceDetectionNode:
    """
    A node that detects faces in an image, crops them, and resizes to 1024x1024 pixels.

    Class methods
    -------------
    INPUT_TYPES (dict):
        Specifies the input parameters of the node.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tuple.
    RETURN_NAMES (`tuple`):
        The name of each output in the output tuple.
    FUNCTION (`str`):
        The name of the entry-point method.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        Returns a dictionary which contains config for all input fields.
        """
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
        # Load the pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_and_crop_faces(self, image):
        """
        Detects faces in the input image, crops them, and resizes to 1024x1024 pixels.

        Parameters:
        image (IMAGE): Input image tensor

        Returns:
        tuple: Contains a tensor of cropped and resized faces
        """
        # Convert the input tensor to a numpy array
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Ensure the image is in the correct format (H, W, C) and uint8
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        image = (image * 255).astype(np.uint8)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(image_rgb)

        cropped_faces = []
        for (x, y, w, h) in faces:
            # Crop the face
            face = image_rgb[y:y+h, x:x+w]
            
            # Resize to 1024x1024
            face_resized = cv2.resize(face, (1024, 1024), interpolation=cv2.INTER_AREA)
            
            # Normalize to 0-1 range and convert to float32
            face_normalized = (face_resized / 255.0).astype(np.float32)
            
            # Convert to tensor and change to channel-first format
            face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1)
            
            cropped_faces.append(face_tensor)

        if not cropped_faces:
            print("No face detected in the image.")
            return (torch.zeros(1, 3, 1024, 1024),)  # Return a blank image if no face is detected
        
        # Stack all cropped faces into a single tensor
        result = torch.stack(cropped_faces)

        # Clear CUDA cache
        torch.cuda.empty_cache()

        return (result,)

    @classmethod
    def IS_CHANGED(s, image):
        return float("NaN")

# This part is for testing the node independently
if __name__ == "__main__":
    # Create a dummy image tensor (you may replace this with actual image loading)
    dummy_image = torch.rand(3, 1000, 1000)

    # Instantiate the node
    face_detector = FaceDetectionNode()

    # Call the node's function
    result = face_detector.detect_and_crop_faces(dummy_image)

    # Print results
    print(f"Number of faces detected: {result[0].shape[0]}")
    print(f"Output shape of each face: {result[0].shape[1:]}")
