from os import path
import numpy as np
import util
import nltk
from nltk.corpus import words
from sklearn.utils import shuffle
import naivebayes
import pdb
# pdb.set_trace()


def main():
    print("-------------PHONEME-------------------")
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

    print("-------------CHAR-------------------")

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
    naivebayes.run_naive_bayes(model, util.get_char_dict(), "saved/baby_gender_predictions_char")

    print("-------------PHONEME + CHAR-------------------")
    matrix = np.concatenate((name_matrix,name_char_matrix),axis=1)
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
    naivebayes.run_naive_bayes(model, util.get_char_dict(), "saved/baby_gender_predictions_char")

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

if __name__ == '__main__':
    main()
