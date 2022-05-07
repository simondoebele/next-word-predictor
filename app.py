import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from torch import layout

from app_components.navbar import navbar
import app_components.n_gram as n_gram
import app_components.neural_networks as nn

from models.n_gram import load
from nltk.tokenize import RegexpTokenizer


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

n_gram_models = {
    n: load(f'./processed_n_grams/{n}-gram.pkl')
    for n in range(1, 6)
}

# Callbacks

## ROuter

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

    if text is None: text = ''

    tokenizer = RegexpTokenizer(r'\w+')
    tokenized = tokenizer.tokenize(text)
    tokens = [word.lower() for word in tokenized]

    # if len(tokens) < n: return '>>> Previous words: ', '>>> First characters: ', '>>> Suggestions: '

    # else:

    #     if text.endswith(' '):

    #         first_characters = ''
    #         words = tokens[- n + 1 :]

    #     else:
            
    #         first_characters = tokens[-1]
    #         words = tokens[- n : -1]
        
    #     predictions = n_gram_models[n].predict(words, first_characters)
    #     predictions = [word for word, frequency in predictions[:10]]

    # if n == 0: words = ''

    if len(tokens) == 0: return '>>> Suggestions: '

    else:

        predictions = []

        for dim in range(1, 6):

            if text.endswith(' '):

                first_characters = ''
                words = tokens[- dim + 1 :]

            else:
                
                first_characters = tokens[-1]
                words = tokens[- dim : -1]

            if dim == 1: words = []

            n_predictions = n_gram_models[dim].predict(words, first_characters)
            n_predictions = [word for word, frequency in n_predictions]
            predictions = n_predictions + predictions

        predictions = list(dict.fromkeys(predictions))    
        
        return f'>>> Suggestions: {str(predictions[:n])}'

if __name__ == '__main__':
    app.run_server(debug=True)