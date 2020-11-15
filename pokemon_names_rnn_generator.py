import csv
import numpy as np
from g2p_en import G2p
import requests
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import LambdaCallback

# Initialize g2p
g2p = G2p()

# Get Pokemon names
result = requests.get("https://pokeapi.co/api/v2/pokemon/?limit=1000").json()
print(result.keys())

full_names = [result['results'][i]['name'] for i in range(min(result['count'],1000))]
print(full_names[:10])

# Fix up some names with -
full_names[121] = 'mistermime'
full_names[249] = 'hooh'
full_names[771] = 'typenull'
full_names[438] = 'mimejunior'
full_names[784] = 'koko'
full_names[785] = 'lele'
full_names[786] = 'bulu'
full_names[787] = 'fini'
full_names[232] = 'porygontwo'
full_names[473] = 'porygonzet'

# Some names are duplicates with - at the end
names_duplicates = list(map(lambda s: s.split('-')[0], full_names))
print(len(names_duplicates))

# Remove duplicates after getting rid of -
names = list(set(names_duplicates))
print(len(names))

# names = list(map(lambda s: s + '.', names))
# print(names[:10])

# Apply g2p to all Pokemon names and add '.' at the end to let the model know the end of each name
names_g2p = list(map(g2p, names))
for phoneme_vec in names_g2p:
    phoneme_vec.append('.')
print(names_g2p[:10])

# Find all phonemes used in the training data
# temp = set()
# for phoneme_vec in names_g2p:
#     temp = temp.union(set(phoneme_vec))
# print(temp)
# print(len(temp))

all_phonemes = sorted(['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'])
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
m = len(names)

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

# texts = ["Korea", "hoonpyo", "amazing!"]
# texts = ["I have $250 in my pocket.", # number -> spell-out
#          "popular pets, e.g. cats and dogs", # e.g. -> for example
#          "I refuse to collect the refuse around here.", # homograph
#          "I'm an activationist."] # newly coined word

# for text in texts:
#     out = g2p(text)
#     print(out)

# with open('yob2019.txt', newline='') as csvfile:
#     namereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     lst_complete = []
#     for row in namereader:
#         temp = []
#         for x in row[0].split(','):
#             temp.append(x)
#         temp[0] = g2p(temp[0])
#         print(temp)
#         lst_complete.append(temp)
#     print(len(lst_complete))



