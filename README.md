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

# Algorithms
Algorithms from class (e.g. logreg, naivebayes, etc.)

Deep Learning (e.g. [RNN](https://towardsdatascience.com/can-data-science-help-you-pick-a-baby-name-b7e98a98268e))

Clustering algorthms (e.g. [k-meansNN](https://arxiv.org/pdf/1808.07292.pdf))

# Datasets
[IMDB Dataset](https://www.kaggle.com/columbine/imdb-dataset-sentiment-analysis-in-csv-format) as "imdb\_train.csv", "imdb\_test.csv", "imdb\_valid.csv"

[Bad Words](https://www.freewebheaders.com/bad-words-list-and-page-moderation-words-list-for-facebook/)

YouTube/Reddit

# Ideas
Detecting personality traits of a person by their writing [Personality Detection through DL](https://link.springer.com/article/10.1007/s10462-019-09770-z)

# Notes from OH
1. In an imbalanced dataset, properly weight the smaller dataset (as in ps1)
2. If you have a long sequence (like IMDB reviews) and you are using RNN, consider pairs of phonemes rather than individual phonemes (simply because the sequences might become really long otherwise).
