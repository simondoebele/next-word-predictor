import string
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from torch import layout

from app_components.navbar import navbar
import app_components.main_page as mainpage

from models.n_gram import load_all
from nltk.tokenize import RegexpTokenizer
from config import *

from app_RNN import load_RNN

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
            children = mainpage.layout
        ),
    ]
)

# Load models

# n_gram_models = load_all(f'processed_n_grams/news20-5.pkl')
n_gram_models = load_all(f'processed_n_grams/5.pkl')
rnn = load_RNN('model_2022-05-19_16_15_44_320110')

# Callbacks

## Router

@app.callback(
    Output('main-page-title', 'children'),
    Input('url', 'pathname')
)

def router(url): 
    if url.__contains__('neural-networks'): return 'Recurrent Neural Network'
    else: return 'n-gram Model'


## List of possible words

@app.callback(
    [
        Output(f'console-predictions-{n}', 'children')
        for n in range(1, 11)
    ],
    Input('main-page-text-area', 'value'),
    Input('main-page-slider', 'value'),
    Input('url', 'pathname')
)

def predict_words(text, n, type): 

    """
    Use of backoff prediction: is n-gram does not exist, check for (n-1)-gram.
    """

    if text is None or text == '': return [None for _ in range(10)]

    if type.__contains__('neural-networks'): use_rnn = True
    else: use_rnn = False

    tokenizer = RegexpTokenizer(r'\w+')
    tokenized = tokenizer.tokenize(text)
    tokens = [word.lower() for word in tokenized]

    if text.endswith(' '):

        first_characters = ''
        words = tuple(tokens[- n_gram_models.dimension + 1 :])

    else:
        
        first_characters = tokens[-1]
        words = tuple(tokens[- n_gram_models.dimension : -1])

    if use_rnn: predictions = rnn.predict_next_word(words, first_characters, n)
    else: predictions = n_gram_models.predict(words, first_characters, n)
    predictions += [None for _ in range(10 - len(predictions))]

    return predictions

## Add word from console

@app.callback(
    Output('main-page-text-area', 'value'),
    [
        Input(f'console-predictions-{n}', 'n_clicks')
        for n in range(1, 11)
    ],
    State('main-page-text-area', 'value'),
    [
        State(f'console-predictions-{n}', 'children')
        for n in range(1, 11)
    ],
    prevent_initial_call = True
)

def add_word(
    click_1, click_2, click_3, click_4, click_5, click_6, click_7, click_8, click_9, click_10,
    text,
    word_1, word_2, word_3, word_4, word_5, word_6, word_7, word_8, word_9, word_10
):

    ctx = dash.callback_context
    triggerer = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if triggerer is None: return text
    else:

        if text.endswith(' '): pass
        elif text is None or text == '': text == ''
        else: 
            while text != '' and text[-1] in string.ascii_lowercase: text = text[:-1]

        if triggerer == 'console-predictions-1': return text + word_1 + ' '
        elif triggerer == 'console-predictions-2': return text + word_2 + ' '
        elif triggerer == 'console-predictions-3': return text + word_3 + ' '
        elif triggerer == 'console-predictions-4': return text + word_4 + ' '
        elif triggerer == 'console-predictions-5': return text + word_5 + ' '
        elif triggerer == 'console-predictions-6': return text + word_6 + ' '
        elif triggerer == 'console-predictions-7': return text + word_7 + ' '
        elif triggerer == 'console-predictions-8': return text + word_8 + ' '
        elif triggerer == 'console-predictions-9': return text + word_9 + ' '
        else: return text + word_10 + ' '

if __name__ == '__main__':
    app.run_server(debug=True)