import random

class RandomSeedGeneratorNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {}
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("Random Seed",)
    FUNCTION = "generate_random_seed"
    CATEGORY = "Generators"

    def generate_random_seed(self):
        seed = random.randint(0, 2**32 - 1)
        return (seed,)
