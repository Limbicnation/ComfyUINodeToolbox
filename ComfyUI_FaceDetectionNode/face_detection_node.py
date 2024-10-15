import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

class FaceDetectionNode:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Cropped Faces",)
    FUNCTION = "detect_and_crop_faces"
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

    def detect_and_crop_faces(self, image):
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
            return (torch.zeros(3, 1024, 1024),)  # Return a blank image if no face is detected
        
        # Stack all cropped faces into a single tensor
        return (torch.stack(cropped_faces),)

    def show_result(self, image_tensors):
        # Convert tensor to numpy array
        image_np = image_tensors.cpu().numpy()
        
        # Plot each face
        num_faces = image_np.shape[0]
        fig, axes = plt.subplots(1, num_faces, figsize=(5*num_faces, 5))
        
        if num_faces == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            face = np.transpose(image_np[i], (1, 2, 0))
            ax.imshow(face)
            ax.axis('off')
            ax.set_title(f"Face {i+1}")
        
        plt.tight_layout()
        plt.show()

# This part is for testing the node independently
if __name__ == "__main__":
    # Create a dummy image tensor (you may replace this with actual image loading)
    dummy_image = torch.rand(3, 1000, 1000)

    # Instantiate the node
    face_detector = FaceDetectionNode()

    # Call the node's function
    result = face_detector.detect_and_crop_faces(dummy_image)

    # Show the result
    face_detector.show_result(result[0])

    print(f"Number of faces detected: {result[0].shape[0]}")
    print(f"Output shape of each face: {result[0].shape[1:]}")
