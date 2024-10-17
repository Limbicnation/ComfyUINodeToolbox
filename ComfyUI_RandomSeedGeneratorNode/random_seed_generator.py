import torch
import numpy as np
import gc

class RandomSeedGeneratorNode:
    """
    A node that generates a random seed for use in other nodes.

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
                "seed_range": ("INT", {
                    "default": 2**32 - 1,
                    "min": 0,
                    "max": 2**32 - 1,
                    "step": 1,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("Random Seed",)
    FUNCTION = "generate_random_seed"
    CATEGORY = "utils"

    def generate_random_seed(self, seed_range):
        """
        Generates a random seed using PyTorch on CPU and sets it for both PyTorch and NumPy.

        Parameters:
        seed_range (int): The maximum value for the generated seed

        Returns:
        tuple: Contains the generated random seed
        """
        seed = 0
        try:
            # Ensure we're using CPU for random number generation
            device = torch.device("cpu")
            with torch.no_grad():
                # Generate random seed on CPU
                seed = torch.randint(0, seed_range, (1,), dtype=torch.int32, device=device).item()
            
            # Set the seed for both PyTorch and NumPy
            torch.manual_seed(seed)
            np.random.seed(seed)

            # If CUDA is available, also set the CUDA seed
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        except Exception as e:
            print(f"An error occurred while generating random seed: {str(e)}")
            seed = 0  # Default to 0 if an error occurs
        finally:
            # Empty torch caches for efficiency
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Perform garbage collection
            gc.collect()
        
        return (seed,)

    @classmethod
    def IS_CHANGED(s, seed_range):
        """
        Ensures that the node always generates a new seed.
        """
        return float("NaN")

# This part is for testing the node independently
if __name__ == "__main__":
    seed_generator = RandomSeedGeneratorNode()
    seed = seed_generator.generate_random_seed(2**32 - 1)[0]
    print(f"Generated Random Seed: {seed}")
