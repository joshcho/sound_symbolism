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
    train_and_predict(True)
    # train_and_predict(False)

def train_and_predict(is_gender_neutral):
    print("-------------PHONEME-------------------")
    phon_dict = util.get_phon_dict()
    data_prefix = "data/names/"
    gen_prefix = "generated/"

    npy_name = ""
    matrix_name = ""
    if is_gender_neutral:
        npy_name = "gender_neutral_names.npy"
        matrix_name = "gender_neutral_name_matrix.gz"
        if not path.exists(gen_prefix + npy_name):
            ds = save_gender_neutral_names_npy(data_prefix, gen_prefix)
        else:
            ds = np.load(gen_prefix + npy_name)
    else:
        npy_name = "names.npy"
        matrix_name = "name_matrix.gz"
        if not path.exists(gen_prefix + npy_name):
            ds = save_names_npy(data_prefix, gen_prefix)
        else:
            ds = np.load(gen_prefix + npy_name)

    ds = np.array(ds)
    name_matrix = np.array([[]])
    if not path.exists(gen_prefix + matrix_name):
        name_matrix = util.transform_text_to_phon_cnts(list(ds[:,0]))
        np.savetxt(gen_prefix + matrix_name, name_matrix)
    else:
        name_matrix = np.loadtxt(gen_prefix + matrix_name)

    # shuffle matrix
    matrix = name_matrix
    labels = ds[:,1].astype('float64')
    matrix, labels = shuffle(matrix, labels, random_state=0)
    labels = np.reshape(labels, (len(labels),))

    model = {}
    idx1 = int(0.8 * len(matrix))
    idx2 = int(0.9 * len(matrix))
    model["train_matrix"] = matrix[0: idx1]
    model["train_labels"] = labels[0: idx1]
    model["val_matrix"] = matrix[idx1: idx2]
    model["val_labels"] = labels[idx1: idx2]
    model["test_matrix"] = matrix[idx2:]
    model["test_labels"] = labels[idx2:]

    naivebayes.run_naive_bayes(model, phon_dict, "saved/baby_gender_predictions")

    print("-------------CHAR-------------------")

    name_char_matrix = util.transform_text_to_char_cnts(list(np.array(ds)[:,0]))
    matrix = name_char_matrix
    labels = ds[:,1].astype('float64')
    matrix, labels = shuffle(matrix, labels, random_state=0)
    labels = np.reshape(labels, (len(labels),))

    model = {}
    idx1 = int(0.8 * len(matrix))
    idx2 = int(0.9 * len(matrix))
    model["train_matrix"] = matrix[0: idx1]
    model["train_labels"] = labels[0: idx1]
    model["val_matrix"] = matrix[idx1: idx2]
    model["val_labels"] = labels[idx1: idx2]
    model["test_matrix"] = matrix[idx2:]
    model["test_labels"] = labels[idx2:]
    naivebayes.run_naive_bayes(model, util.get_char_dict(), "saved/baby_gender_predictions_char")

    print("-------------PHONEME + CHAR-------------------")
    matrix = np.concatenate((name_matrix,name_char_matrix),axis=1)
    labels = ds[:,1].astype('float64')
    matrix, labels = shuffle(matrix, labels, random_state=0)
    labels = np.reshape(labels, (len(labels),))

    model = {}
    idx1 = int(0.8 * len(matrix))
    idx2 = int(0.9 * len(matrix))
    model["train_matrix"] = matrix[0: idx1]
    model["train_labels"] = labels[0: idx1]
    model["val_matrix"] = matrix[idx1: idx2]
    model["val_labels"] = labels[idx1: idx2]
    model["test_matrix"] = matrix[idx2:]
    model["test_labels"] = labels[idx2:]
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

def save_gender_neutral_names_npy(data_prefix, gen_prefix):
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

    gender_neutral_names = boy_names.intersection(girl_names)
    non_gender_neutral_names = boy_names.union(girl_names) - gender_neutral_names

    ds = []
    for name in gender_neutral_names:
        ds.append([name, 1])
    index = 0
    for name in non_gender_neutral_names:
        index +=1
        ds.append([name, 0])
        if index > len(gender_neutral_names):
            break
    np.save(gen_prefix + "gender_neutral_names.npy", ds)
    return ds

if __name__ == '__main__':
    main()
