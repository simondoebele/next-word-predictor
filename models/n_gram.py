from collections import defaultdict
from typing import List, Tuple
from nltk import FreqDist
from nltk.util import ngrams 
from nltk.tokenize import RegexpTokenizer
import pickle

import os
import numpy as np
from time import time

FILES = ['./data/' + filename for filename in os.listdir('./data') if filename.startswith('H')]

class FilterableDict(FreqDist):

    def __init__(self, n_gram_model: int, samples = None):
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

def generate(n_max = 5):

    models = {
        i: load(f'processed_n_grams/news20-{i}-gram.pkl') for i in range(1, n_max + 1)
    }

    d = defaultdict(list)

    for n in range(n_max, 0, -1):

        for key in models[n]:
            d[key[:-1]].append((key[-1], models[n][key]))

    file = open(f"processed_n_grams/news20-{n_max}.pkl", "wb")
    pickle.dump([*d.items()], file)
    file.close()

    print('File generated')

def load_all(filename: str):

    with open(filename, 'rb') as file: data = pickle.load(file)

    return nGram(list, dict(data))

class nGram(defaultdict):

    def __init__(self, default, dictionnary):
        super().__init__(default, dictionnary)
        self.dimension = max([len(key) for key in self.keys()]) + 1

    def predict(self, words: tuple, first_characters: str = '', limit: int = 4) -> List[Tuple[str, int]]:
        
        predictions = []

        for n in range(0, self.dimension):

            filtered = list(
                filter(
                    lambda item: item[0].startswith(first_characters),
                    self[words[n:]]
                )
            )

            filtered.sort(key = lambda y: y[1], reverse = True)
            sublist = [key for key, value in filtered]

            predictions += sublist

        return list(dict.fromkeys(predictions))[:limit]

    
#generate()

def process_news(n_min, n_max):

    t = time()

    for n in range(n_min, n_max + 1):
        model = nGramProcessor(n, ['data/news20.txt'])
        model.save(f'processed_n_grams/news20-{n}-gram.pkl')
        print(f'{n}-gram processed in {time() - t} seconds.')
        t = time()
    
    generate()

#process_news(1,5)
#generate(4)