from os import path
import numpy as np
import util
import nltk
from nltk.corpus import words
from sklearn.utils import shuffle
import naivebayes

def main():
    phon_dict = util.get_phon_dict()

    model = train_bad_words()
    naivebayes.run_naive_bayes(model, phon_dict, "saved/bad_words_predictions")

    model = train_imdb()
    naivebayes.run_naive_bayes(model, phon_dict, "saved/imdb_predictions")

def train_imdb():
    train_texts, train_labels = util.load_csv('data/imdb_train.csv')
    val_texts, val_labels = util.load_csv('data/imdb_valid.csv')
    test_texts, test_labels = util.load_csv('data/imdb_test.csv')
    train_m_path = "saved/imdb_train_matrix.gz"
    val_m_path = "saved/imdb_val_matrix.gz"
    test_m_path = "saved/imdb_test_matrix.gz"

    if path.exists(train_m_path):
        train_matrix = np.loadtxt(train_m_path)
    else:
        train_matrix = util.transform_text_to_phon_cnts(train_texts)
        np.savetxt(train_m_path, train_matrix)

    if path.exists(val_m_path):
        val_matrix = np.loadtxt(val_m_path)
    else:
        val_matrix = util.transform_text_to_phon_cnts(val_texts)
        np.savetxt(val_m_path, val_matrix)

    if path.exists(test_m_path):
        test_matrix = np.loadtxt(test_m_path)
    else:
        test_matrix = util.transform_text_to_phon_cnts(test_texts)
        np.savetxt(test_m_path, test_matrix)

    model = {}
    model["train_matrix"] = train_matrix
    model["val_matrix"] = val_matrix
    model["test_matrix"] = test_matrix
    model["train_labels"] = train_labels
    model["val_labels"] = val_labels
    model["test_labels"] = test_labels
    return model

def train_bad_words():
    nltk.download('words')
    nltk_words = words.words()
    nltk_matrix_path = "saved/nltk_words_matrix.gz"
    bad_matrix_path = "saved/fb_bad_words_matrix.gz"

    # Note that some words in bad_words may not be in nltk_words
    bad_words = np.loadtxt("data/facebook-bad-words.txt",delimiter=',',dtype=str)

    # Load if already saved
    if path.exists(nltk_matrix_path):
        nltk_matrix = np.loadtxt(nltk_matrix_path)
    else:
        nltk_matrix = util.transform_text_to_phon_cnts(word_set)
        np.savetxt(nltk_matrix_path, nltk_matrix)

    if path.exists(bad_matrix_path):
        bad_matrix = np.loadtxt(bad_matrix_path)
    else:
        bad_matrix = util.transform_text_to_phon_cnts(word_set)
        np.savetxt(bad_matrix_path, bad_matrix)

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
    return model

if __name__ == '__main__':
    main()
