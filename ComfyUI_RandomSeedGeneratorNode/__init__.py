from .random_seed_generator import RandomSeedGeneratorNode

NODE_CLASS_MAPPINGS = {
    "RandomSeedGenerator": RandomSeedGeneratorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomSeedGenerator": "Random Seed Generator"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
