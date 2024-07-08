import torch
from transformers import CLIPTokenizer, CLIPTextModel
from typing import Tuple, Dict, List

class ImprovedClipTextEncoderNode:
    MAX_LENGTH = 77

    def __init__(self):
        """
        Initializes the tokenizer and model for the text encoder.
        """
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model or tokenizer: {e}")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple[str, Dict[str, int]]]]:
        """
        Returns the required input types for the encode_text function.
        """
        return {
            "required": {
                "text_input": ("STRING", {"default": "Enter your text here"}),
                "max_length": ("INT", {"default": cls.MAX_LENGTH, "min": 1, "max": cls.MAX_LENGTH}),
            },
        }

    RETURN_TYPES = ("CLIP", "CONDITIONING")
    RETURN_NAMES = ("clip_output", "conditioning_output")
    FUNCTION = "encode_text"
    CATEGORY = "Improved"

    def encode_text(self, text_input: str, max_length: int = MAX_LENGTH) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Encodes the input text using the CLIP model and returns the embeddings.
        
        Parameters:
        text_input (str): The input text to encode.
        max_length (int): The maximum length for tokenization.

        Returns:
        Tuple: A tuple containing the clip output and conditioning output.
        """
        if not isinstance(text_input, str) or not text_input.strip():
            raise ValueError("Invalid text input. It must be a non-empty string.")
        
        if not isinstance(max_length, int) or not (1 <= max_length <= self.MAX_LENGTH):
            raise ValueError(f"Invalid max_length. It must be an integer between 1 and {self.MAX_LENGTH}.")

        try:
            # Tokenize the input text
            inputs = self.tokenizer(
                text_input, 
                padding="max_length", 
                max_length=max_length, 
                truncation=True, 
                return_tensors="pt"
            )

            # Generate text embeddings using the CLIP model
            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                pooled_output = outputs.pooler_output

            # Prepare the CLIP and CONDITIONING outputs
            clip_output = {
                "text_embeddings": last_hidden_state,
                "pooled_output": pooled_output
            }
            conditioning_output = [[last_hidden_state, {"pooled_output": pooled_output}]]

            return clip_output, conditioning_output

        except Exception as e:
            raise RuntimeError(f"Error during text encoding: {e}")

# Register the node
NODE_CLASS_MAPPINGS = {
    "ImprovedClipTextEncoder": ImprovedClipTextEncoderNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImprovedClipTextEncoder": "Improved CLIP Text Encoder"
}
