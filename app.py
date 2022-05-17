import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from torch import layout

from app_components.navbar import navbar
import app_components.n_gram as n_gram
import app_components.neural_networks as nn

from models.n_gram import load_all
from nltk.tokenize import RegexpTokenizer
from config import *


app = dash.Dash(
    __name__,
    external_stylesheets = [
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
        'https://fonts.googleapis.com/css2?family=Secular+One&family=Special+Elite&display=swap',
    ]
)

server = app.server
app.config.suppress_callback_exceptions = True

app.title = "Word Predictor"
app.layout = html.Div(
    [
        navbar,
        html.Br(),
        dcc.Location(id = 'url', refresh = False),
        dbc.Container(
            id = 'main-page',
            children = n_gram.layout
        ),
    ]
)

# Load models

n_gram_models = load_all(f'processed_n_grams/news-2-gram.pkl')

# Callbacks

## Router

@app.callback(
    Output('main-page', 'children'),
    Input('url', 'pathname')
)

def router(url): 
    if url.__contains__('neural-networks'): return nn.layout
    else: return n_gram.layout


## List of possible words

@app.callback(
    Output('console-predictions', 'children'),
    Input('n-gram-text-area', 'value'),
    Input('n-gram-slider', 'value')
)

def predict_words(text, n): 

    """
    Use of backoff prediction: is n-gram does not exist, check for (n-1)-gram.
    """

    if text is None or text == '': return '>>> Suggestions: '

    tokenizer = RegexpTokenizer(r'\w+')
    tokenized = tokenizer.tokenize(text)
    tokens = [word.lower() for word in tokenized]

    if text.endswith(' '):

        first_characters = ''
        words = tuple(tokens[- n_gram_models.dimension + 1 :])

    else:
        
        first_characters = tokens[-1]
        words = tuple(tokens[- n_gram_models.dimension : -1])

    return f'>>> Suggestions: {str(n_gram_models.predict(words, first_characters, n))}'

    # predictions = []

    # for dim in range(N_GRAM, 0, -1):

    #     if text.endswith(' '):

    #         first_characters = ''
    #         words = tokens[- dim + 1 :]

    #     else:
            
    #         first_characters = tokens[-1]
    #         words = tokens[- dim : -1]

    #     if dim == 1: words = []

    #     n_predictions = n_gram_models[dim].predict(words, first_characters)[:n - len(predictions)]
    #     n_predictions = [word for word, frequency in n_predictions]
    #     predictions += n_predictions

    #     if len(predictions) == n: break

    # predictions = list(dict.fromkeys(predictions))    
    
    # return f'>>> Suggestions: {str(predictions)}'

if __name__ == '__main__':
    app.run_server(debug=True)