import dash_bootstrap_components as dbc

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("n-Gram", href="n-gram", style = {'font-family': 'Segoe UI'})),
        dbc.NavItem(dbc.NavLink("Neural Networks", href="neural-networks", style = {'font-family': 'Segoe UI'})),
    ],
    brand="WordPredictor.",
    brand_href="#",
    color="#065168",
    dark=True,
    style = {'font-family': 'Special Elite'}
)