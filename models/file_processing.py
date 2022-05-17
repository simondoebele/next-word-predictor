from typing import List
from nltk.tokenize import RegexpTokenizer

def process(filenames: List[str]) -> List[str]:

    tokenizer = RegexpTokenizer(r'\w+')

    for filename in filenames:

            print(f'Processing {filename}')
            textfile = open(filename, 'r', errors='ignore')

            for line in textfile:
                    
                tokenized = tokenizer.tokenize(line)
                tokens = [word.lower() for word in tokenized]

            if len(tokens) > 0: yield tokens

            textfile.close()