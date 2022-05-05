import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

layout = [
    dbc.Row(
        html.H2(
            'Neural Networks',
            style = {'font-family': 'Secular One'}
        ),
    ),
    html.Br(),
    dbc.Row(
        dbc.Textarea(
            id = 'nn-text-area',
            placeholder = 'Type something...'
        )
    )
]