from .naive_bayes_node import NaiveBayesNode

NODE_CLASS_MAPPINGS = {
    "NaiveBayesNode": NaiveBayesNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NaiveBayesNode": "Naive Bayes Probability Node"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
