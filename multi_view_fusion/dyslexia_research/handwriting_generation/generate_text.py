import random
import re
import nltk
nltk.download('brown')
nltk.download('punkt')  # needed for tokenization if we do custom work
from nltk.corpus import brown

def apply_inconsistent_spacing(text, spacing_probability=0.3):
    words = text.split()
    result_words = []
    for word in words:
        if random.random() < spacing_probability:
            # Apply inconsistent spacing to this word
            result_words.append(word + ' ' * random.choice([1, 2, 3]))
        else:
            result_words.append(word + ' ')
    return ''.join(result_words).strip() # .strip() to remove trailing space

def apply_mirror_letters(text):
    mirror_map = {'b': 'd', 'd': 'b', 'p': 'q', 'q': 'p'}
    return ''.join([mirror_map.get(char, char) for char in text])

def apply_random_capitals(text):
    target_letters = {'s', 'k', 't', 'l'}
    return ''.join([
        char.upper() if char.lower() in target_letters and random.random() < 0.3 else char
        for char in text
    ])

common_misspellings = {
    'because': ['becuase', 'becaus', 'becasue'],
    'was': ['wos', 'waz'],
    'allow': ['alolow'],
    'square': ['squar'],
    'friend': ['fiernd', 'firend','frend'],
    'Wednesday': ['Wensday'],
    'phone': ['fone'],
    'remember': ['remeber'],
    'coming': ['comming'],
    'their': ["they're", 'there'],
    'museum': ['musem'],
    'likes': ['liks'],
    'to':['too', 'two'],
    'teal': ['teel'],
    'has':['haz'],
    'beach':['bech'],
    'shopping':['shoping'],
    'black':['blak', 'blac'],
    'restaurant':['restrant'],
    'yellow':['yello'],
    'maroon':['marun'],
    'play':['paly'],
    'mouse':['mous'],
    'purple':['purpl'],
    'rabbit':['rabit']
}

def apply_spelling_errors(text):
    for word, misspellings in common_misspellings.items():
        if word in text:
            # Use regex with word boundaries to replace whole words only
            text = re.sub(r'\b' + re.escape(word) + r'\b', random.choice(misspellings), text, flags=re.IGNORECASE)
    return text

def degrade_grammar(text):
    # This function is designed to introduce grammatical errors.
    # It currently lowers the first letter and removes the full stop.
    # This function can be extended to include more diverse grammatical degradation.
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]
    if text.endswith('.'):
        text = text[:-1]
    return text

# Define probabilities for each effect
EFFECT_PROBABILITIES = {
    'spelling_errors': 0.8,
    'mirror_letters': 0.3,
    'random_capitals': 0.54,
    'inconsistent_spacing': 0.3,
    'degrade_grammar':0.5,
}

def generate_dyslexic_sentence(base_sentence):
    """
    Generates a sentence with dyslexic traits from the base_sentence.

    Args:
        base_sentence : Sentence to be modified.

    Returns:
        string :  A sentence with dyslexic traits.
    """
    s = base_sentence

    if random.random() < EFFECT_PROBABILITIES['spelling_errors']:
        s = apply_spelling_errors(s)
    if random.random() < EFFECT_PROBABILITIES['mirror_letters']:
        s = apply_mirror_letters(s)
    if random.random() < EFFECT_PROBABILITIES['random_capitals']:
        s = apply_random_capitals(s)
    if random.random() < EFFECT_PROBABILITIES['degrade_grammar']:
        s = degrade_grammar(s)
    if random.random() < EFFECT_PROBABILITIES['inconsistent_spacing']:
        s = apply_inconsistent_spacing(s)

    return s

'''sentence = "I have a friend called Bob, he's very kind."
print("Original:", sentence)
print("Dyslexic:", generate_dyslexic_sentence(sentence))

# Example with another sentence
sentence2 = "The quick brown fox jumps over the lazy dog."
print("\nOriginal:", sentence2)
print("Dyslexic:", generate_dyslexic_sentence(sentence2))'''

def generate_sentences(num_sentences=500):
    """
    Generates a specified number of sentences with random names, colours, pets, and places.

    Args:
        num_sentences (int): The number of sentences to generate.

    Returns:
        list: A list of generated sentences.
    """

    # Define lists of random elements
    names = [
        "Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "George", "Hannah",
        "Isaac", "Jasmine", "Kevin", "Laura", "Michael", "Nora", "Oliver", "Penny",
        "Quinn", "Rachel", "Sam", "Tina", "Umar", "Violet", "William", "Jennifer",
        "Luca", "Zack", "Mia", "Leo", "Chloe", "Noah", "Sophia", "Liam", "Olivia",
        "Emma", "Ava", "Isabella", "Jackson", "Lucas", "Aiden", "Harper", "Evelyn",
        "Abigail", "Emily", "Ella", "Scarlett", "Grace", "Lily", "Zoe", "Riley", "Florence"
    ]

    colours = [
        "red", "blue", "green", "yellow", "orange", "purple", "pink", "black",
        "white", "brown", "grey", "gold", "silver", "teal", "maroon", "navy",
        "beige", "turquoise", "lavender", "indigo"
    ]

    pets = [
        "dog", "cat", "rabbit", "hamster", "guinea pig", "parrot", "fish",
        "snake", "turtle", "ferret", "lizard", "mouse", "rat", "chinchilla",
        "hedgehog", "gerbil", "canary", "budgie", "gecko", "axolotl"
    ]

    places = [
        "park", "library", "cafe", "museum", "beach", "mountains", "forest",
        "zoo", "cinema", "shopping mall", "gym", "swimming pool", "art gallery",
        "concert hall", "bookstore", "restaurant", "playground", "lake",
        "river", "stadium"
    ]

    ways = ["play at", "visit", "go to", "take a trip to"]

    generated_sentences = []

    for _ in range(num_sentences):
        # Randomly select one item from each list
        name = random.choice(names)
        colour = random.choice(colours)
        pet = random.choice(pets)
        place = random.choice(places)
        how = random.choice(ways)

        # Construct the sentences using f-strings for easy formatting
        sentence1 = f"My best friend, {name}, has a {colour} pet {pet}."
        sentence2 = f"{name} likes to {how} the {place}."

        generated_sentences.append(sentence1 + " " + sentence2)

    return generated_sentences


