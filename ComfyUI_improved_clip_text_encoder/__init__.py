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

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "RandomSeedGenerator": RandomSeedGeneratorNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomSeedGenerator": "Random Seed Generator"
}
