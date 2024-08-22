import random
from typing import Tuple, List, Optional

class RandomSeedGeneratorNode:
    """Generates random seeds for other nodes."""
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("Random Seed",)
    FUNCTION = "generate_random_seed"
    CATEGORY = "Generators"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """Defines the input types for the node."""
        return {
            "required": {}
        }

    def generate_random_seed(self) -> Tuple[int]:
        """Generates a random seed between 0 and 2**32 - 1."""
        seed = random.randint(0, 2**32 - 1)
        return (seed,)


class KSamplerNode:
    """Performs sampling using specified models and parameters."""
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "sample"
    CATEGORY = "Samplers"

    DEFAULT_STEPS = 20
    DEFAULT_CFG = 8.0
    DEFAULT_SAMPLER_NAME = 'euler'
    DEFAULT_SCHEDULER = 'normal'
    DEFAULT_DENOISE = 1.0

    def __init__(self):
        self.model = None
        self.positive = None
        self.negative = None
        self.latent_image = None
        self.seed = None
        self.steps = self.DEFAULT_STEPS
        self.cfg = self.DEFAULT_CFG
        self.sampler_name = self.DEFAULT_SAMPLER_NAME
        self.scheduler = self.DEFAULT_SCHEDULER
        self.denoise = self.DEFAULT_DENOISE

    @classmethod
    def INPUT_TYPES(cls):
        """Defines the input types for the node."""
        return {
            "required": {
                "model": ("LATENT",),
                "positive": ("TEXT",),
                "negative": ("TEXT",),
                "latent_image": ("LATENT",),
                "seed": ("INT",),
                "steps": ("INT",),
                "cfg": ("FLOAT",),
                "sampler_name": ("TEXT",),
                "scheduler": ("TEXT",),
                "denoise": ("FLOAT",)
            }
        }

    def sample(
        self,
        model: Optional[any] = None,
        positive: Optional[str] = "",
        negative: Optional[str] = "",
        latent_image: Optional[any] = None,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        cfg: Optional[float] = None,
        sampler_name: Optional[str] = None,
        scheduler: Optional[str] = None,
        denoise: Optional[float] = None
    ) -> Tuple[List[any]]:
        """Performs the sampling using the provided parameters."""
        # Use the seed provided to initialize the random generator
        if seed is not None:
            random.seed(seed)

        # Placeholder for actual sampling logic
        samples = []  # Replace with actual implementation
        
        return (samples,)


# Usage Example
if __name__ == "__main__":
    seed_generator = RandomSeedGeneratorNode()
    
    # Generate a new seed for each iteration
    for _ in range(5):  # Simulate multiple runs
        seed = seed_generator.generate_random_seed()[0]

        ksampler = KSamplerNode()
        samples = ksampler.sample(
            model=None, positive="", negative="", latent_image=None, seed=seed,
            steps=20, cfg=8.0, sampler_name='euler', scheduler='normal', denoise=1.0
        )
        print("Generated Seed:", seed)
        print("Generated Samples:", samples)
