# Setup
```
pip3 install g2p_en
python g2p.py
```
See [g2p](https://github.com/Kyubyong/g2p) for details.
# Research Referents
[LSTM/RNN for Baby Names](https://towardsdatascience.com/can-data-science-help-you-pick-a-baby-name-b7e98a98268e)

[Brand Names](https://www.nickkolenda.com/brand-names/)

[Meaning to Form](https://www.aclweb.org/anthology/P19-1171.pdf)

[Critique of Meaning to Form](https://medium.com/@rmalouf/measuring-systematicity-aa562e73f7af)

[Nominative Determinism](https://en.wikipedia.org/wiki/Nominative_determinism)

[YouTube Topic ID](https://gist.github.com/stpe/2951130dfc8f1d0d1a2ad736bef3b703)

# Algorithms
Algorithms from class (e.g. logreg, naivebayes, etc.)

Deep Learning (e.g. [Phoneme-based RNN for Baby Names](https://towardsdatascience.com/can-data-science-help-you-pick-a-baby-name-b7e98a98268e), [Phoneme-based RNN for Word Identification](http://papers.neurips.cc/paper/372-a-recurrent-neural-network-for-word-identification-from-continuous-phoneme-strings.pdf))

Clustering algorthms (e.g. [k-meansNN](https://arxiv.org/pdf/1808.07292.pdf))

Phonestheme2vec (e.g. [Investigation into Phonesthemes](https://www.aclweb.org/anthology/N16-1038.pdf), [Phn2vec embeddings](https://bootphon.blogspot.com/2014/05/phn2vec-embeddings.html), [Phoneme embeddings](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1037&context=scil))

[Transfer Learning](https://www.coursera.org/learn/convolutional-neural-networks/lecture/4THzO/transfer-learning)

# Datasets
[IMDB Dataset](https://www.kaggle.com/columbine/imdb-dataset-sentiment-analysis-in-csv-format) as "imdb\_train.csv", "imdb\_test.csv", "imdb\_valid.csv"

[Bad Words](https://www.freewebheaders.com/bad-words-list-and-page-moderation-words-list-for-facebook/)

YouTube/Reddit

# Ideas
- Detecting personality traits of a person by their writing [Personality Detection through DL](https://link.springer.com/article/10.1007/s10462-019-09770-z)
- Phonestheme-based RNN (This is genuinely novel)
  - Create phonestheme embedddings (e.g. [word2vec](https://jalammar.github.io/illustrated-word2vec/), [phonestheme discovery](https://www.aclweb.org/anthology/W18-1206/)). My hypothesis is that this will work much better than phoneme embeddings. We may only train on subsets of language that are likely to be sound symbolic, like *names*.
  - Predict things based on phonestheme embeddings using RNN on channel names, etc.
  - This is similar to learning embeddings for OOV (out-of-vocabulary) words like in [fasttext](https://datascience.stackexchange.com/questions/54806/word-embedding-of-a-new-word-which-was-not-in-training)
- Learn the word embedding of the closest word, and use it along with phoneme/phonestheme embeddings.

# Notes from OH
1. In an imbalanced dataset, properly weight the smaller dataset (as in ps1)
2. If you have a long sequence (like IMDB reviews) and you are using RNN, consider pairs of phonemes rather than individual phonemes (simply because the sequences might become really long otherwise).
