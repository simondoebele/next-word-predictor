import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

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
    n: load(f'./processed_n_grams/{n}-gram')
    for n in range(2, 11)
}

# Callbacks

@app.callback(
    Output('n-gram-title', 'children'),
    Input('n-gram-slider', 'value')
)

def update_title(n): return f'{n}-Gram Model'

## List of possible words

@app.callback(
    Output('console-previous-words', 'children'),
    Output('console-first-characters', 'children'),
    Output('console-predictions', 'children'),
    Input('n-gram-text-area', 'value'),
    Input('n-gram-slider', 'value')
)

def predict_words(text, n): 

    if text is None: text = ''

    tokenizer = RegexpTokenizer(r'\w+')
    tokenized = tokenizer.tokenize(text)
    tokens = [word.lower() for word in tokenized]

    if len(tokens) < n: return '>>> Previous words: ', '>>> First characters: ', '>>> Suggestions: '

    else:

        if text.endswith(' '):

            first_characters = ''
            words = tokens[- n + 1 :]

        else:
            
            first_characters = tokens[-1]
            words = tokens[- n : -1]
        
        predictions = n_gram_models[n].predict(words, first_characters)
        predictions = [word for word, frequency in predictions[:10]]
    
    return f'>>> Previous words: {str(words)}', f'>>> First characters: {first_characters}', f'>>> Suggestions: {str(predictions)}'

if __name__ == '__main__':
    app.run_server(debug=True)