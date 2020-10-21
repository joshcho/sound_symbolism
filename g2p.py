from g2p_en import G2p
import numpy as np

g2p = G2p()

texts = ["I have $250 in my pocket.", # number -> spell-out
         "popular pets, e.g. cats and dogs", # e.g. -> for example
         "I refuse to collect the refuse around here.", # homograph
         "I'm an activationist.", "I"] # newly coined word
# This is a list of list of phonemes, e.g. [["AY1", " ", "HH", ...], ["P","AA1", ...],...].
pseqs = [g2p(text) for text in texts]

def create_pdict(texts):
    """
    Args:
        texts: Texts as input

    Return:
        Dictionary with a phoneme as key and index as value
    """
    all_phonemes = set()
    for pseq in pseqs:
        for p in pseq:
            all_phonemes.add(p)    
    
    pdict = {}
    index = 0
    for p in all_phonemes:
        pdict[p] = index
        index += 1
    return pdict

def create_pvecs(pseqs, pdict):
    """
    Args:
        pseqs: A list of phoneme sequences. It ends up being a list of list of phonemes. Every list of phonemes is a g2p conversion from a text.
        pdict: A dictionary with keys as a phoneme and value as index.

    Return:
        List of vectors of phoneme counts
    """
    pvecs = []
    for pseq in pseqs:
        pvec = [0]*len(pdict)
        for p in pseq:
            pvec[pdict[p]] += 1
        pvecs.append(pvec)
    return pvecs

pdict = create_pdict(texts)
print(len(pdict))

pvecs = create_pvecs(pseqs, pdict)
print(pvecs)
