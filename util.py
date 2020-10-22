import bz2
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

def load_csv(csv_path):
    """Load the spam dataset from a TSV file

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
