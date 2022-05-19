from collections import defaultdict
import pickle
from typing import Tuple
from app_RNN import RNN, load_RNN
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def evaluate(model: RNN, test_file: str, max_recommendation: int = 4) -> Tuple[dict, dict]:

    # Step 1: Initialising the dictionnaries 

    evaluation = {
        recommendations : 0 for recommendations in range(1, max_recommendation + 1)
    }

    # Step 2: Reading the file

    textfile = open(test_file, 'r', errors='ignore')
    tokenized_lines = []

    for line in textfile:
            
        # Remove non-letters characters
        tokenizer = RegexpTokenizer(r'\w+')
        tokenized = tokenizer.tokenize(line)
        tokens = [word.lower() for word in tokenized]
        if len(tokens) > 1: tokenized_lines.append(tokens[:]) # if only one word, not interesting

    textfile.close()

    # Step 3: Evaluation

    keystrokes_to_save = 0

    for line in tqdm(tokenized_lines):

        line_length = len(line)
        for ind in range(1, line_length - 1):

            previous_words, word_to_predict = tuple(line[:ind]), line[ind]
            length = len(word_to_predict)
            keystrokes_to_save += length

            done = [False for _ in range(max_recommendation)]

            for chars in range(length):

                predictions = model.predict_next_word(previous_words, word_to_predict[:chars], max_recommendation)
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

def evaluate_and_save(model: RNN, test_file: str, max_recommendation: int = 4) -> None:

    evaluation = evaluate(model, test_file, max_recommendation)

    file = open("evaluation/rnn/saved_keystrokes.pkl", "wb")
    pickle.dump(evaluation, file)
    file.close()

def load(filename: str = "evaluation/rnn/saved_keystrokes.pkl"):

    with open(filename, 'rb') as file: data = pickle.load(file)
    return dict(data)

def plot_graph(data: dict) -> None:

    recommendations = len(data.keys())
    x_axis = np.arange(1, recommendations + 1)
    y_axis = list(data.values())

    colors = ['#fee721', '#3fbc71', '#335e8d']
    width = 0.3

    #plt.bar(x_axis, y_axis, width = width, color = colors[2])
    plt.plot(x_axis, y_axis, marker = 'o', linestyle = 'dashed', color = colors[2])

    plt.xlabel("Number of recommendations")
    plt.ylabel("Saved keystrokes (%)")

    plt.title('Evaluation of the RNN')

    plt.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)

    plt.xticks(x_axis) 
    plt.ylim(bottom=0, top=100)

    plt.legend()

    plt.savefig('evaluation/rnn/saved_keystrokes.png')
    plt.show()

# model = load_all('processed_n_grams/news20-5.pkl')
model = load_RNN('model_2022-05-19_16_15_44_320110')
evaluate_and_save(model, 'data/test.txt', 5)

dic = load("evaluation/rnn/saved_keystrokes.pkl")
plot_graph(dic)