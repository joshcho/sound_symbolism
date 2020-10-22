from g2p_en import G2p
import numpy as np
import util
import naivebayes
import os.path
from os import path
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

def split_words(all_words):
    random.seed(1234)
    train_words = \
        random.sample(all_words, len(all_words) *2 // 3)
    diff_words = list(set(all_words).difference(set(train_words)))
    val_words = random.sample(diff_words, len(diff_words) // 2)
    test_words = list(set(diff_words).difference(set(val_words)))
    return train_words, val_words, test_words

def transform_words_and_save(word_set, save_path):
    matrix = transform_text(word_set, phoneme_dict)
    np.savetxt(save_path, matrix)

def train_bad_words():
    nltk_matrix_path = "nltk_words_matrix.gz"
    bad_matrix_path = "fb_bad_words_matrix.gz"
    nltk.download('words')
    nltk_words = words.words()

    # Note that some words in bad_words may not be in nltk_words
    bad_words = np.loadtxt("facebook-bad-words.txt",delimiter=',',dtype=str)

    if not path.exists(nltk_matrix_path):
        transform_words_and_save(nltk_words, nltk_matrix_path)

    if not path.exists(bad_matrix_path):
        transform_words_and_save(bad_words, bad_matrix_path)

    nltk_matrix = np.loadtxt(nltk_matrix_path)
    bad_matrix = np.loadtxt(bad_matrix_path)

    if False: # analyze frequency of words in each dataset
        nltk_counts = np.sum(nltk_matrix,axis=0)
        bad_counts = np.sum(bad_matrix,axis=0)
        # freq = bad_counts/np.sum(bad_counts)
        freq = nltk_counts/np.sum(nltk_counts)
        freq_dict = phoneme_dict.copy()
        for phoneme, i in freq_dict.items():
            freq_dict[phoneme] = freq[i]

        some_list = []
        for phoneme in sorted(freq_dict, key=freq_dict.get,reverse=True):
            some_list.append((phoneme, freq_dict[phoneme]))
        print(some_list)
        return

    nltk_labels = np.array([[1 if word in bad_words else 0 \
                             for word in nltk_words]]).T
    bad_labels = np.ones((len(bad_words), 1))

    # shuffling both matrix and labels together and merging nltk and bad
    nltk_dummy = np.concatenate((nltk_matrix, nltk_labels), axis=1)
    bad_dummy = np.concatenate((bad_matrix, bad_labels), axis=1)
    dummy = np.concatenate((nltk_dummy, bad_dummy), axis=0)
    np.random.shuffle(dummy)
    matrix, labels = np.split(dummy, [-1], axis=1)
    labels = np.reshape(labels, (len(labels),))

    model = {}
    n = len(matrix)
    index1 = n // 3
    index2 = n * 2 // 3
    model["train_matrix"] = matrix[0: index1]
    model["train_labels"] = labels[0: index1]
    model["val_matrix"] = matrix[index1: index2]
    model["val_labels"] = labels[index1: index2]
    model["test_matrix"] = matrix[index2:]
    model["test_labels"] = labels[index2:]

    naivebayes.run_naive_bayes(model, phoneme_dict, "bad_words_predictions")

train_bad_words()

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
        val_matrix = transform_text(val_texts, phoneme_dict)
        np.savetxt("imdb_val_matrix.gz", val_matrix)
        test_matrix = transform_text(test_texts, phoneme_dict)
        np.savetxt("imdb_test_matrix.gz", test_matrix)
    train_matrix = np.loadtxt("imdb_train_matrix.gz")
    val_matrix = np.loadtxt("imdb_val_matrix.gz")
    test_matrix = np.loadtxt("imdb_test_matrix.gz")


    model = {}
    model["train_matrix"] = train_matrix
    model["val_matrix"] = val_matrix
    model["test_matrix"] = test_matrix
    model["train_labels"] = train_labels
    model["val_labels"] = val_labels
    model["test_labels"] = test_labels

    naivebayes.run_naive_bayes(model, phoneme_dict, "imdb_predictions")
