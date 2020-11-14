from os import path
import numpy as np
import util
import nltk
from nltk.corpus import words
from sklearn.utils import shuffle
import naivebayes
import pdb
# pdb.set_trace()

def save_names_npy(data_prefix, gen_prefix):
    boy_names = set()
    girl_names = set()
    for yr in range(1880,2020):
        yob_path = data_prefix + "yob" + str(yr) + ".txt"
        baby_names = np.loadtxt(yob_path, delimiter=',',dtype=str)
        for entry in baby_names:
            name, gender, _ = entry
            if gender == "M":
                boy_names.add(name)
            else:
                girl_names.add(name)

    ds = []
    for name in boy_names:
        ds.append([name, 1])
    for name in girl_names:
        ds.append([name, 0])
    np.save(gen_prefix + "names.npy", ds)
    return ds

def main():
    phon_dict = util.get_phon_dict()
    data_prefix = "data/names/"
    gen_prefix = "generated/"

    ds = []
    if not path.exists(gen_prefix + "names.npy"):
        ds = save_names_npy(data_prefix, gen_prefix)
    else:
        ds = np.load(gen_prefix + "names.npy")

    name_matrix = np.array([[]])
    if not path.exists(gen_prefix + "name_matrix.gz"):
        name_matrix = util.transform_text_to_phon_cnts(list(np.array(ds)[:,0]))
        np.savetxt(gen_prefix + "name_matrix.gz", name_matrix)
    else:
        name_matrix = np.loadtxt(gen_prefix + "name_matrix.gz")

    # shuffle matrix
    matrix = name_matrix
    labels = ds[:,1].astype('float64')
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

    naivebayes.run_naive_bayes(model, phon_dict, "saved/baby_gender_predictions")

    name_char_matrix = util.transform_text_to_char_cnts(list(np.array(ds)[:,0]))
    matrix = name_char_matrix
    labels = ds[:,1].astype('float64')
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
    print("-------------CHAR-------------------")
    naivebayes.run_naive_bayes(model, util.get_char_dict(), "saved/baby_gender_predictions_char")


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
