from typing import List, Tuple
from nltk import FreqDist
from nltk.util import ngrams 
from nltk.tokenize import RegexpTokenizer
import pickle

import os

import numpy as np

FILES = ['./data/' + filename for filename in os.listdir('./data') if filename.startswith('H')]

class FilterableDict(FreqDist):

    def __init__(self, n_gram_model: int, samples=None):
        super().__init__(samples)
        self.n_gram_model = n_gram_model

    def predict(self, words: List[str], first_characters: str = '') -> List[Tuple[str, int]]:

        if len(words) != self.n_gram_model - 1:
            return []

        filtered = list(
            filter(
                lambda item: (words == list(item[0])[:-1] and item[0][-1].startswith(first_characters)),
                self.items()
            )
        )

        #fast_filtered = np.array(filtered, dtype = [('word', 'S10'), ('occurencies', int)]).sort(order = 'occurencies')

        sublist = [(key[-1], value) for key, value in filtered]
        sublist.sort(key = lambda y: y[1], reverse = True)

        return sublist

    def save(self, filename: str) -> None:

        file = open(f"{filename}", "wb")
        pickle.dump([*self.items()], file)
        file.close()

        print(f'Data saved in {filename}')

class nGramProcessor():

    def __init__(self, n: int, filenames: List[str]) -> None:
        
        self.n = n
        self.frequencies = FilterableDict(n)

        # Process the files

        for filename in filenames:

            print(f'Processing {filename}')
            textfile = open(filename, 'r', errors='ignore')

            for line in textfile:

                if len(line) > 1:
                    
                    # Remove non-letters characters
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokenized = tokenizer.tokenize(line)
                    tokens = [word.lower() for word in tokenized]

                    n_grams = ngrams(tokens, n)
                    self.frequencies.update(n_grams)

            textfile.close()

    def save(self, filename: str) -> None: self.frequencies.save(filename)

def load(filename: str) -> FilterableDict:

    with open(filename, 'rb') as file: data = pickle.load(file)

    n_gram_model = len(data[0][0])    
    return FilterableDict(n_gram_model, dict(data))


# model = nGramProcessor(1, FILES)
# model.save(f'processed_n_grams/{1}-gram.pkl')

# a = np.array([('hi', 1), ('how', 2), ('are', 3), ('you', 1)], dtype = [('word', 'S10'), ('occurencies', int)])
# fil = lambda item, words, first_characters: (words == list(item[0])[:-1] and item[0][-1].startswith(first_characters))
# print(fil(a, [], 'h'))

# data = load('./processed_n_grams/3-gram.pkl')
# print(data.predict(['harry', 'potter'], ''))
# print(data.predict(['on', 'the'], 'mas'))
# print(data.predict(['house', 'of']))