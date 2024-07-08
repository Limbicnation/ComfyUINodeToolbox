# naive_bayes_node.py

# Probabilities extracted from the image
accessory_prob = {
    "OhneBrille": 0.22,
    "RundeBrille": 0.44,
    "Sonnenbrille": 0.34
}

clothing_prob = {
    "Kaputzenpullover": 0.14,
    "Kittel": 0.36,
    "HemdSchaufelausschnitt": 0.38,
    "HemdVausschnitt": 0.12
}

color_prob = {
    "Rot": 0.46,
    "Blau": 0.54  # Assuming other values, since blue is not given in the table
}

hair_prob = {
    "Langhaar": 0.5,  # Assuming equal distribution
    "Kurzhaar": 0.5   # Assuming equal distribution
}

clothing_color_prob = {
    "Weiss": 0.44,
    "Blau": 0.56  # Assuming other values, since blue is not given in the table
}

def compute_naive_bayes(hair, color, accessory, clothing, clothing_color):
    prob = (hair_prob[hair] *
            color_prob[color] *
            accessory_prob[accessory] *
            clothing_prob[clothing] *
            clothing_color_prob[clothing_color])
    return prob

class NaiveBayesNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hair": (["Langhaar", "Kurzhaar"], {"default": "Langhaar"}),
                "color": (["Rot", "Blau"], {"default": "Rot"}),
                "accessory": (["OhneBrille", "RundeBrille", "Sonnenbrille"], {"default": "RundeBrille"}),
                "clothing": (["Kaputzenpullover", "Kittel", "HemdSchaufelausschnitt", "HemdVausschnitt"], {"default": "HemdSchaufelausschnitt"}),
                "clothing_color": (["Weiss", "Blau"], {"default": "Weiss"})
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("probability",)
    FUNCTION = "compute_probability"
    CATEGORY = "Probability Computation"

    def compute_probability(self, hair, color, accessory, clothing, clothing_color):
        return (compute_naive_bayes(hair, color, accessory, clothing, clothing_color),)

NODE_CLASS_MAPPINGS = {
    "NaiveBayesNode": NaiveBayesNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NaiveBayesNode": "Naive Bayes Probability Node"
}
