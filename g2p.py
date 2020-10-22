from g2p_en import G2p
import numpy as np
import util
import time as ti

sample_texts = ["I have $250 in my pocket.", # number -> spell-out
         "popular pets, e.g. cats and dogs", # e.g. -> for example
         "I refuse to collect the refuse around here.", # homograph
         "I'm an activationist.", "I"] # newly coined word

g2p = G2p()

def time(function, variable, iterations):
    tic = ti.perf_counter()
    for i in range(iterations):
        function(variable)
    toc = ti.perf_counter()
    fname = str(function)
    if hasattr(function, '__name__'):
        fname = function.__name__
    print(f"{iterations} iterations of {fname} ran in {toc - tic:0.4f} seconds")

all_phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + \
    ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2',
        'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2',
        'B', 'CH', 'D', 'DH','EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2',
        'EY0', 'EY1','EY2', 'F', 'G', 'HH','IH0', 'IH1', 'IH2', 'IY0',
        'IY1', 'IY2', 'JH', 'K', 'L','M', 'N', 'NG', 'OW0', 'OW1','OW2',
        'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH','UH0', 'UH1',
        'UH2', 'UW','UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']

def generate_phoneme_dict():
    index = 0
    pdict = {}
    for phoneme in all_phonemes:
        pdict[phoneme] = index
        index += 1
    return pdict

def transform_text(texts, phoneme_dict):
    """
    Args:
        texts: A list of strings where each string is a text.
        pdict: A dictionary with keys as a phoneme and value as index.

    Return:
        List of vectors of phoneme counts
    """
    matrix = []
    index = 0
    for text in texts:
        arr = [0] * len(phoneme_dict)
        for phoneme in g2p(text):
            if phoneme in phoneme_dict:
                arr[phoneme_dict[phoneme]] += 1
        matrix.append(arr)
        if index % 100 == 0:
            print("%d/%d texts transformed" % (index, len(texts)))
        index += 1
    return np.array(matrix)

train_texts, train_labels = \
    util.load_csv('imdb_train.csv')
val_texts, val_labels = \
    util.load_csv('imdb_valid.csv')
test_texts, test_labels = \
    util.load_csv('imdb_test.csv')

phoneme_dict = generate_phoneme_dict()

train_matrix = transform_text(train_texts, phoneme_dict)
np.savetxt("train_matrix.gz", train_matrix)
print(train_matrix)
print(train_matrix.shape)
