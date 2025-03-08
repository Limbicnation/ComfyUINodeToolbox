import random
import numpy as np
import torch

class RandomSeedGeneratorNode:
    """
    A node that generates seed values for consistent or random image generation.
    
    Features:
    - Multiple seed modes (random, fixed)
    - Cross-library seed synchronization (random, numpy, torch)
    - GPU-compatible with CUDA support
    - Optional deterministic mode for complete reproducibility
    - State tracking for debugging
    """
    def __init__(self):
        self.last_mode = None
        self.last_seed = None
        self.deterministic = False
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["random", "fixed"],),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**32 - 1,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "sync_libraries": ("BOOLEAN", {"default": True}),
                "deterministic": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "generate_seed"
    CATEGORY = "utils/random"
    
    def generate_seed(self, mode, seed, sync_libraries=True, deterministic=False):
        """
        Generate a seed value based on the selected mode.
        
        Args:
            mode (str): 'random' for new random seed each time, 'fixed' for user-defined seed
            seed (int): User-defined seed value (used when mode is 'fixed')
            sync_libraries (bool): Whether to synchronize seed across random, numpy, and torch
            deterministic (bool): Enable full deterministic mode (may impact performance)
            
        Returns:
            tuple(int,): Generated seed value
        """
        # Store deterministic setting
        self.deterministic = deterministic
        
        if mode == "random":
            # Generate new random seed
            generated_seed = random.randint(0, 2**32 - 1)
            self.last_seed = generated_seed
            self.last_mode = mode
            
            # Apply the seed if synchronization is enabled
            if sync_libraries:
                self._apply_seed(generated_seed, deterministic)
                
            return (generated_seed,)
        else:  # fixed mode
            # Use user-provided seed
            self.last_seed = seed
            self.last_mode = mode
            
            # Apply the seed if synchronization is enabled
            if sync_libraries:
                self._apply_seed(seed, deterministic)
                
            return (seed,)
    
    def _apply_seed(self, seed_value, deterministic=False):
        """
        Apply the seed value across multiple libraries for consistent randomization.
        
        Args:
            seed_value (int): The seed value to apply
            deterministic (bool): Whether to enable full deterministic mode
        """
        # Set Python's random module seed
        random.seed(seed_value)
        
        # Set NumPy seed
        np.random.seed(seed_value)
        
        # Set PyTorch seeds
        torch.manual_seed(seed_value)
        
        # Set CUDA seed if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
        
        # Enable deterministic mode if requested
        if deterministic and torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        elif not deterministic and torch.cuda.is_available():
            # Reset to performance defaults when not using deterministic mode
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    
    def reset(self):
        """
        Reset the node's state to default values.
        """
        self.last_mode = None
        self.last_seed = None
        self.deterministic = False
    
    @classmethod
    def IS_CHANGED(cls, mode, seed, sync_libraries=True, deterministic=False):
        """
        Ensure the node is re-executed when in random mode.
        
        Returns:
            Union[float, bool]: Random value to force re-execution in random mode, False otherwise
        """
        if mode == "random":
            # Return random value to force re-execution
            return random.random()
        return False