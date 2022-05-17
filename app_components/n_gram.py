import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from config import *

layout = [

    # Page title

    dbc.Row(
        html.H2(
            children = 'n-gram Model',
            id = 'n-gram-title',
            style = {'font-family': 'Secular One'}
        ),
    ),

    html.Br(),

    # Slider to select the model

    dcc.Slider(
        min=1,
        max=10,
        step=1,
        value=DEFAULT_RECOMMENDATIONS,
        id='n-gram-slider',
        disabled=False,
        dots=True,
        marks={ i: {'label': str(i)} for i in range(1, 16) },
    ),

    html.Br(),

    # Text area for the inputs

    dbc.Textarea(
        id = 'n-gram-text-area',
        placeholder = 'Type something...',
    ),

    html.Br(),

    # Console to display the results

    dbc.Card(
        dbc.CardBody(
            [ 
                html.P(
                    [
                        html.I(className='fa fa-code'),
                        html.Strong('   Console')
                    ]
                ),
                # html.P('>>> Previous words: ', id='console-previous-words'),
                # html.P('>>> First characters: ', id='console-first-characters'),
                html.P('>>> Suggestions: ', id='console-predictions')
            ]
        ),
        style = {'font-family': 'consolas', 'background-color': '#f2f2f2'} 
    )
]