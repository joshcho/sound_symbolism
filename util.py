import csv
import pandas as pd
import time as ti
import numpy as np
from g2p_en import G2p

all_phons = ["<pad>", "<unk>", "<s>", "</s>"] + \
    ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2',
        'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2',
        'B', 'CH', 'D', 'DH','EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2',
        'EY0', 'EY1','EY2', 'F', 'G', 'HH','IH0', 'IH1', 'IH2', 'IY0',
        'IY1', 'IY2', 'JH', 'K', 'L','M', 'N', 'NG', 'OW0', 'OW1','OW2',
        'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH','UH0', 'UH1',
        'UH2', 'UW','UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']

def get_phon_dict():
    index = 0
    phon_dict = {}
    for phon in all_phons:
        phon_dict[phon] = index
        index += 1
    return phon_dict

def transform_text_to_phon_cnts(texts):
    """
    Args:
        texts: A list of strings where each string is a text.

    Return:
        List of vectors of phoneme counts
    """
    g2p = G2p()
    phon_dict = get_phon_dict()
    matrix = []
    index = 0
    for text in texts:
        arr = [0] * len(phon_dict)
        for phon in g2p(text):
            if phon in phon_dict:
                arr[phon_dict[phon]] += 1
        matrix.append(arr)
        if index % 100 == 0:
            print("%d/%d texts transformed" % (index, len(texts)))
        index += 1
    return np.array(matrix)

def load_csv(csv_path):
    """Load the dataset from a TSV file

    Args:
         csv_path: Path to TSV file containing dataset.

    Returns:
        texts: A list of string values containing the text of each message.
        labels: The binary labels (0 or 1) for each message. A 1 indicates positive sentiment.
    """
    data = pd.read_csv(csv_path,sep=',',header=0).values
    texts = data[:,0].tolist()
    labels = data[:,1]

    return texts, labels

def time(function, variable, iterations):
    tic = ti.perf_counter()
    for i in range(iterations):
        function(variable)
    toc = ti.perf_counter()
    fname = str(function)
    if hasattr(function, '__name__'):
        fname = function.__name__
    print(f"{iterations} iterations of {fname} ran in {toc - tic:0.4f} seconds")
