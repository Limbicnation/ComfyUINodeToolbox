import random

class RandomSeedGeneratorNode:
    """
    A node that generates seed values for consistent or random image generation.
    
    Modes:
    - random: Generates a new random seed for each execution
    - fixed: Uses the user-provided seed value
    """
    def __init__(self):
        self.last_mode = None
        self.last_seed = None

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
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "generate_seed"
    CATEGORY = "utils/random"

    def generate_seed(self, mode, seed):
        """
        Generate a seed value based on the selected mode.
        
        Args:
            mode (str): 'random' for new random seed each time, 'fixed' for user-defined seed
            seed (int): User-defined seed value (used when mode is 'fixed')
            
        Returns:
            tuple(int,): Generated seed value
        """
        if mode == "random":
            # Generate new random seed
            generated_seed = random.randint(0, 2**32 - 1)
            self.last_seed = generated_seed
            self.last_mode = mode
            return (generated_seed,)
        else:  # fixed mode
            # Use user-provided seed
            self.last_seed = seed
            self.last_mode = mode
            return (seed,)

    @classmethod
    def IS_CHANGED(cls, mode, seed):
        """
        Ensure the node is re-executed when in random mode.
        """
        if mode == "random":
            # Return current timestamp to force re-execution
            return random.random()
        return False
