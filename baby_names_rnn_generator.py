import csv
import numpy as np
from g2p_en import G2p
import requests
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import LambdaCallback
from collections import Counter

# Initialize g2p
g2p = G2p()

# Year information
start_year = 1880
finish_year = 2019

# Dictionaries to store names
names_by_year = dict() # keep names by year
gender_neutral_names_by_year = dict() # keep gender neutral names by year
num_names_by_year = dict() # keep total number of names by year

# Get baby names for a given year
for year in range(start_year,finish_year+1):
    with open('baby_names/yob{}.txt'.format(year)) as f:
        lines = f.readlines()
    names = list(map(lambda s: s.split(',')[0], lines)) # raw names
    
    # Identify the names that appeared more than once
    # Each name only appears at most twice, for each gender
    # Within each gender, there is no duplicate names
    name_counter = Counter(names) # make a counter of baby names
    gender_neutral_names = []
    for k in name_counter.keys():
        if name_counter[k] > 1:
            gender_neutral_names.append(k)
    gender_neutral_names_by_year[year] = gender_neutral_names

    # Store names
    num_names_by_year[year] = (len(names), len(set(names)), len(names)-len(set(names)), (len(names)-len(set(names)))/len(names)*100) # add total number of names
    names_by_year[year] = set(names) # remove duplicates and add to the names dictionary

# Data Exploration
# print(num_names_by_year)
# temp = []
# for year in num_names_by_year.keys():
#     temp.append(num_names_by_year[year][3])
# print(sum(temp)/len(temp))

# Apply g2p to all baby names and add '.' at the end to let the model know the end of each name
names_g2p = list(map(g2p, list(names_by_year[finish_year])))
for phoneme_vec in names_g2p:
    phoneme_vec.append('.')
print(names_g2p[:10])

# Find all phonemes used in the training data
all_phonemes = set()
for phoneme_vec in names_g2p:
    all_phonemes = all_phonemes.union(set(phoneme_vec))
print(all_phonemes)
print(len(all_phonemes))

all_phonemes = sorted(list(all_phonemes))
all_phonemes.remove('.')
num_phonemes = len(all_phonemes)
phoneme_to_index = dict()
index_to_phoneme = dict()

for i in range(num_phonemes):
    phoneme_to_index[all_phonemes[i]] = i+1
    phoneme_to_index[' '] = 0
    phoneme_to_index['.'] = num_phonemes + 1

    index_to_phoneme[i+1] = all_phonemes[i]
    index_to_phoneme[0] = ' '
    index_to_phoneme[num_phonemes + 1] = '.'

print(phoneme_to_index)
print(index_to_phoneme)

# Sanity check
# for k in phoneme_to_index.keys():
#     print(index_to_phoneme[phoneme_to_index[k]] == k)

# Convert from char to index
# char_to_index = dict( (chr(i+96), i) for i in range(1,27))
# char_to_index[' '] = 0
# char_to_index['.'] = 27

# Convert from index to phoneme
# index_to_char = dict( (i, chr(i+96)) for i in range(1,27))
# index_to_char[0] = ' '
# index_to_char[27] = '.'

# maximum number of phonemes in Pokémon names
# this will be the number of time steps in the RNN
max_phonemes = len(max(names_g2p, key=len))
print(max_phonemes)

# number of elements in the list of names, this is the number of training examples
m = len(names_g2p)

# number of potential characters, this is the length of the input of each of the RNN units
dim_phonemes = len(phoneme_to_index)

# Initialize X & Y
X = np.zeros((m, max_phonemes, dim_phonemes))
Y = np.zeros((m, max_phonemes, dim_phonemes))

# Transform phoneme vectors into index vectors
for i in range(m):
    name = names_g2p[i]
    for j in range(len(name)):
        X[i, j, phoneme_to_index[name[j]]] = 1
        if j < len(name)-1:
            Y[i, j, phoneme_to_index[name[j+1]]] = 1

# Make the model
model = Sequential()
model.add(LSTM(128, input_shape=(max_phonemes, dim_phonemes), return_sequences=True))
model.add(Dense(dim_phonemes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Make phoneme name
def make_name(model):
    name = []
    x = np.zeros((1, max_phonemes, dim_phonemes))
    end = False
    i = 0
    
    while end==False:
        probs = list(model.predict(x)[0,i])
        probs = probs / np.sum(probs)
        index = np.random.choice(range(dim_phonemes), p=probs)
        if i == max_phonemes-2:
            phoneme = '.'
            end = True
        else:
            phoneme = index_to_phoneme[index]
        name.append(phoneme)
        x[0, i+1, index] = 1
        i += 1
        if phoneme == '.':
            end = True
    
    print(name)

def generate_name_loop(epoch, _):
    if epoch % 25 == 0:
        
        print('Phoneme names generated after epoch %d:' % epoch)

        for _ in range(3):
            make_name(model)
        
        print()

name_generator = LambdaCallback(on_epoch_end = generate_name_loop)

model.fit(X, Y, batch_size=64, epochs=300, callbacks=[name_generator], verbose=0)

# Make names using model
for _ in range(20):
    make_name(model)

# End of code
print("Build succeeded!")


