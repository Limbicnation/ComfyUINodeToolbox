import torch
from transformers import T5EncoderModel, T5Tokenizer

class CLIPTextEncodeFlux:
    """
    A Flux-enhanced CLIP text encoder node that combines CLIP and T5XXL models
    for improved text understanding and conditioning generation.
    """
    def __init__(self):
        self.t5_model = None
        self.t5_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "clip_l": ("STRING", {
                    "multiline": True,
                    "default": "Enter CLIP text here"
                }),
                "t5xxl": ("STRING", {
                    "multiline": True,
                    "default": "Enter T5XXL text here"
                }),
                "guidance": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "flux_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "flux_mode": (["balanced", "creative", "precise"],),
                "semantic_weight": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                })
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/text"

    def load_t5_model(self):
        if self.t5_model is None:
            self.t5_model = T5EncoderModel.from_pretrained("t5-xxl").to(self.device)
            self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-xxl")

    def process_with_flux(self, clip_embed, t5_embed, flux_strength, flux_mode, semantic_weight):
        # Normalize embeddings
        clip_norm = torch.nn.functional.normalize(clip_embed, p=2, dim=-1)
        t5_norm = torch.nn.functional.normalize(t5_embed, p=2, dim=-1)
        
        # Apply flux mode adjustments
        if flux_mode == "balanced":
            flux_matrix = torch.lerp(clip_norm, t5_norm, semantic_weight)
        elif flux_mode == "creative":
            # Add controlled randomness for creative variation
            noise = torch.randn_like(clip_norm) * 0.1
            flux_matrix = torch.lerp(clip_norm + noise, t5_norm, semantic_weight)
        else:  # precise mode
            # Enhance semantic alignment
            attention = torch.matmul(clip_norm, t5_norm.transpose(-2, -1))
            attention = torch.softmax(attention / torch.sqrt(torch.tensor(clip_norm.size(-1))), dim=-1)
            flux_matrix = torch.matmul(attention, t5_norm)
        
        # Apply flux strength
        flux_matrix = flux_matrix * flux_strength
        
        return flux_matrix

    def encode(self, clip, clip_l, t5xxl, guidance, flux_strength, flux_mode, semantic_weight):
        """
        Encodes text inputs using both CLIP and T5XXL models with Flux enhancement.
        
        Parameters:
            clip (CLIP): CLIP model object
            clip_l (str): Text input for CLIP encoding
            t5xxl (str): Text input for T5XXL encoding
            guidance (float): Generation guidance scale
            flux_strength (float): Intensity of Flux processing
            flux_mode (str): Processing mode (balanced/creative/precise)
            semantic_weight (float): Balance between CLIP and T5XXL influence
            
        Returns:
            tuple: Contains the enhanced conditioning output
        """
        # Load T5 model if not loaded
        self.load_t5_model()
        
        # Process CLIP text
        tokens = clip.tokenize(clip_l)
        clip_embed = clip.encode_from_tokens(tokens)
        
        # Process T5XXL text
        t5_tokens = self.t5_tokenizer(t5xxl, return_tensors="pt", padding=True, truncation=True).to(self.device)
        t5_embed = self.t5_model(**t5_tokens).last_hidden_state
        
        # Apply Flux processing
        flux_enhanced = self.process_with_flux(clip_embed, t5_embed, flux_strength, flux_mode, semantic_weight)
        
        # Combine with guidance scale
        cond = flux_enhanced * guidance
        
        # Create final conditioning
        return ([[cond, {"pooled_output": clip_embed.pooled}]], )
