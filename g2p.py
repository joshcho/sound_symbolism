from g2p_en import G2p

# citation
# @misc{g2pE2019,
#   author = {Park, Kyubyong & Kim, Jongseok},
#   title = {g2pE},
#   year = {2019},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   howpublished = {\url{https://github.com/Kyubyong/g2p}}
# }

texts = ["I have $250 in my pocket.", # number -> spell-out
         "popular pets, e.g. cats and dogs", # e.g. -> for example
         "I refuse to collect the refuse around here.", # homograph
         "I'm an activationist."] # newly coined word
g2p = G2p()
for text in texts:
    out = g2p(text)
    print(out)
