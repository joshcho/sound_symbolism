from g2p_en import G2p
import numpy as np
import util
import naivebayes
# for bad_words
import nltk
from nltk.corpus import words
import random
"""
Feature List
1. Save phonemes, not encodings. We may use the phonemes later for RNN.
2. If we need to do a lot of conversions, then directly using cmudict in nltk.corpus may be better than using g2p. g2p predicts pronunciation for words that do not exist, which may take a lot of time. For something like YouTube channel names, however, g2p is very useful.
"""

g2p = G2p()
phoneme_dict = util.get_phoneme_dict()

def transform_text(texts, phoneme_dict):
    """
    Args:
        texts: A list of strings where each string is a text.
        phoneme_dict: A dictionary with keys as a phoneme and value as index.

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

def train_bad_words():
    nltk.download('words')
    random.seed(1234)
    bad_words = np.loadtxt("bad-words.txt",dtype=str)
    # some bad words are not in nltk corpus
    all_words = list(set(random.sample(words.words(),len(bad_words)*5)).union(bad_words))
    train_words = \
        random.sample(all_words, len(all_words) * 3 // 4)
    test_words = list(set(all_words).difference(set(train_words)))
    train_labels = np.array([1 if word in bad_words else 0 \
                             for word in train_words])
    test_labels = np.array([1 if word in bad_words else 0 \
                             for word in test_words])

    train_matrix = transform_text(train_words, phoneme_dict)
    np.savetxt("bad_words_train_matrix.gz", train_matrix)
    test_matrix = transform_text(test_words, phoneme_dict)
    np.savetxt("bad_words_test_matrix.gz", test_matrix)

    # train_matrix = np.loadtxt("bad_words_train_matrix.gz")
    # test_matrix = np.loadtxt("bad_words_test_matrix.gz")

    naivebayes.run_naive_bayes(train_matrix, train_labels,
                               test_matrix, test_labels,
                               phoneme_dict, "bad_words_predictions")

def train_imdb():
    train_texts, train_labels = \
        util.load_csv('imdb_train.csv')
    val_texts, val_labels = \
        util.load_csv('imdb_valid.csv')
    test_texts, test_labels = \
        util.load_csv('imdb_test.csv')

    if False:
        train_matrix = transform_text(train_texts, phoneme_dict)
        np.savetxt("imdb_train_matrix.gz", train_matrix)
        test_matrix = transform_text(test_texts, phoneme_dict)
        np.savetxt("imdb_test_matrix.gz", test_matrix)
    print(len(test_labels))
    print(np.sum(test_labels))

    train_matrix = np.loadtxt("imdb_train_matrix.gz")
    test_matrix = np.loadtxt("imdb_test_matrix.gz")
    naivebayes.run_naive_bayes(train_matrix, train_labels,
                               test_matrix, test_labels,
                               phoneme_dict, "imdb_predictions")
