from collections import defaultdict
import pickle
from typing import Tuple
from n_gram import nGram, load_all
from nltk.util import ngrams 
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def evaluate(model: nGram, test_file: str, max_recommendation: int = 4) -> Tuple[dict, dict]:

    # Step 1: Initialising the dictionnaries 

    evaluation = {
        recommendations : 0 for recommendations in range(1, max_recommendation + 1)
    }

    # Step 2: Reading the file

    textfile = open(test_file, 'r', errors='ignore')
    n_grams = []

    for line in textfile:
            
        # Remove non-letters characters
        tokenizer = RegexpTokenizer(r'\w+')
        tokenized = tokenizer.tokenize(line)
        tokens = [word.lower() for word in tokenized]

        n_grams += ngrams(tokens, model.dimension)

    textfile.close()

    # Step 3: Evaluation

    keystrokes_to_save = 0

    for n_gram in tqdm(n_grams):

        previous_words, word_to_predict = n_gram[:-1], n_gram[-1]
        length = len(word_to_predict)
        keystrokes_to_save += length

        done = [False for _ in range(max_recommendation)]

        for chars in range(length):

            predictions = model.predict(previous_words, word_to_predict[:chars], max_recommendation)
            n_predictions = len(predictions)

            for recommendation in range(1, n_predictions + 1):
                if not done[recommendation - 1]:
                    if word_to_predict in predictions[:recommendation]:
                        evaluation[recommendation] += length - chars # saved keystrokes
                        done[recommendation - 1] = True

            if np.all(done): break

    evaluation = {
        recommendations : 100 * (evaluation[recommendations] / keystrokes_to_save) for recommendations in range(1, max_recommendation + 1)
    }

    return evaluation

def evaluate_and_save(model: nGram, test_file: str, max_recommendation: int = 4) -> None:

    evaluation = evaluate(model, test_file, max_recommendation)

    file = open("saved_keystrokes.pkl", "wb")
    pickle.dump(evaluation, file)
    file.close()

def load(filename: str = "saved_keystrokes.pkl"):

    with open(filename, 'rb') as file: data = pickle.load(file)
    return dict(data)

def bar_chart(data: dict) -> None:

    recommendations = len(data.keys())
    x_axis = np.arange(1, recommendations + 1)
    y_axis = list(data.values())

    colors = ['#fee721', '#3fbc71', '#335e8d']
    width = 0.3

    #plt.bar(x_axis, y_axis, width = width, color = colors[2])
    plt.plot(x_axis, y_axis, marker = 'o', linestyle = 'dashed', color = colors[2])

    plt.xlabel("Number of recommendations")
    plt.ylabel("Saved keystrokes (%)")

    plt.title('Evaluation of the n-gram model')

    plt.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)

    plt.xticks(x_axis) 
    plt.ylim(bottom=0, top=100)

    plt.legend()

    plt.savefig('barchart_ngram.png')
    plt.show()

# model = load_all('processed_n_grams/news20-5.pkl')
# model = load_all('processed_n_grams/data.pkl')
# evaluate_and_save(model, 'data/test.txt', 5)

dic = load("saved_keystrokes.pkl")
bar_chart(dic)