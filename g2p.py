from g2p_en import G2p
import numpy as np
import util
import naivebayes
import os.path
from os import path
import matplotlib.pyplot as plt
# for bad_words
import nltk
from nltk.corpus import words
import random
from sklearn.utils import shuffle

g2p = G2p()
phoneme_dict = util.get_phoneme_dict()
nltk_matrix_path = "nltk_words_matrix.gz"
bad_matrix_path = "fb_bad_words_matrix.gz"

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

def transform_words_and_save(word_set, save_path):
    matrix = transform_text(word_set, phoneme_dict)
    np.savetxt(save_path, matrix)

def train_bad_words():
    nltk.download('words')
    nltk_words = words.words()

    # Note that some words in bad_words may not be in nltk_words
    bad_words = np.loadtxt("facebook-bad-words.txt",delimiter=',',dtype=str)

    # Load if already saved
    if path.exists(nltk_matrix_path):
        nltk_matrix = np.loadtxt(nltk_matrix_path)
    else:
        transform_words_and_save(nltk_words, nltk_matrix_path)

    if path.exists(bad_matrix_path):
        bad_matrix = np.loadtxt(bad_matrix_path)
    else:
        transform_words_and_save(bad_words, bad_matrix_path)

    nltk_labels = np.array([[1 if word in bad_words else 0 \
                             for word in nltk_words]]).T
    bad_labels = np.ones((len(bad_words), 1))

    # shuffle matrix
    matrix = np.concatenate((nltk_matrix, bad_matrix), axis=0)
    labels = np.concatenate((nltk_labels, bad_labels), axis=0)
    matrix, labels = shuffle(matrix, labels, random_state=0)
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
    train_texts, train_labels = util.load_csv('imdb_train.csv')
    val_texts, val_labels = util.load_csv('imdb_valid.csv')
    test_texts, test_labels = util.load_csv('imdb_test.csv')
    train_m_path = "imdb_train_matrix.gz"
    val_m_path = "imdb_val_matrix.gz"
    test_m_path = "imdb_test_matrix.gz"

    if path.exists(train_m_path):
        train_matrix = np.loadtxt(train_m_path)
    else:
        train_matrix = transform_text(train_texts, phoneme_dict)
        np.savetxt(train_m_path, train_matrix)

    if path.exists(val_m_path):
        val_matrix = np.loadtxt(val_m_path)
    else:
        val_matrix = transform_text(val_texts, phoneme_dict)
        np.savetxt(val_m_path, val_matrix)

    if path.exists(test_m_path):
        test_matrix = np.loadtxt(test_m_path)
    else:
        test_matrix = transform_text(test_texts, phoneme_dict)
        np.savetxt(test_m_path, test_matrix)

    model = {}
    model["train_matrix"] = train_matrix
    model["val_matrix"] = val_matrix
    model["test_matrix"] = test_matrix
    model["train_labels"] = train_labels
    model["val_labels"] = val_labels
    model["test_labels"] = test_labels

    naivebayes.run_naive_bayes(model, phoneme_dict, "imdb_predictions")
