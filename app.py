import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from app_components.navbar import navbar
import app_components.n_gram as n_gram
import app_components.neural_networks as nn


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

# Callbacks

@app.callback(
    Output('n-gram-title', 'children'),
    Input('n-gram-slider', 'value')
)

def update_title(n): return f'{n}-Gram Model'

if __name__ == '__main__':
    app.run_server(debug=True)