from g2p_en import G2p
import numpy as np
import util

g2p = G2p()

texts = ["I have $250 in my pocket.", # number -> spell-out
         "popular pets, e.g. cats and dogs", # e.g. -> for example
         "I refuse to collect the refuse around here.", # homograph
         "I'm an activationist.", "I"] # newly coined word
text_phoneme_cache = {}

def get_phonemes(text):
    if text in text_phoneme_cache:
        return text_phoneme_cache[text]
    else:
        phonemes = g2p(text)
        text_phoneme_cache[text] = phonemes
        return phonemes

def create_phoneme_dict(texts):
    """
    Args:
        texts: Texts as input

    Return:
        Dictionary with a phoneme as key and index as value
    """
    all_phonemes = set()
    for text in texts:
        for phoneme in get_phonemes(text):
            all_phonemes.add(phoneme)

    phoneme_dict = {}
    index = 0
    for phoneme in all_phonemes:
        phoneme_dict[phoneme] = index
        index += 1
    return phoneme_dict

def transform_text(texts, phoneme_dict):
    """
    Args:
        texts: A list of strings where each string is a text.
        pdict: A dictionary with keys as a phoneme and value as index.

    Return:
        List of vectors of phoneme counts
    """
    matrix = []
    for text in texts:
        arr = [0] * len(phoneme_dict)
        for phoneme in get_phonemes(text):
            if phoneme in phoneme_dict:
                arr[phoneme_dict[phoneme]] += 1
        matrix.append(arr)
    return np.array(matrix)

phoneme_dict = create_phoneme_dict(texts)
print(len(phoneme_dict))

train_matrix = transform_text(texts, phoneme_dict)
print(train_matrix)

train_texts, train_labels = \
    util.load_csv('imdb_train.csv')
val_texts, val_labels = \
    util.load_csv('imdb_valid.csv')
test_texts, test_labels = \
    util.load_csv('imdb_test.csv')
