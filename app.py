# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, dcc, html, dash_table, Input, Output
import plotly.express as px
import pandas as pd
from sqlalchemy import create_engine, text
from pymongo import MongoClient
from neo4j import GraphDatabase
from flask_caching import Cache
import dash_bootstrap_components as dbc

CACHE_TIMEOUT_SECONDS = 300

app = Dash(
    name=__name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

cache = Cache(
    app.server,
    config={
        # try 'filesystem' if you don't want to setup redis
        'CACHE_TYPE': 'FileSystemCache',
        'CACHE_DIR': './cache/',
    }
)

# app.config.suppress_callback_exceptions = True


def df_to_dash_data_table(df):
    return [df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]]


class Neo4J:

    def __init__(self):
        driver = GraphDatabase.driver("bolt://localhost:7687/academicworld", auth=("neo4j", "t817"))
        session = driver.session(database="academicworld")
        self._session = session

    def query(self, query_str):
        result = self._session.run(query_str)
        df = pd.DataFrame.from_records(result.data())
        return df


class MongoDB:

    def __init__(self):
        conn = MongoClient('mongodb://xzhu:t817@localhost:27017/academicworld?authSource=admin')
        self._db = conn['academicworld']

    def query(self, query_fn):
        cursor = query_fn(self._db)
        df = pd.DataFrame(list(cursor))
        return df


class MySQL:

    def __init__(self):
        engine = create_engine("mysql+pymysql://xzhu:t817@localhost/AcademicWorld", echo=True, future=True)
        conn = engine.connect()
        self._conn = conn

    def query(self, query_str, params=[]):
        result = pd.read_sql(text(query_str), self._conn, params=params)
        return result


mysql = MySQL()
neo4j = Neo4J()
mongodb = MongoDB()


@cache.memoize(timeout=CACHE_TIMEOUT_SECONDS)
def get_most_popular_keywords(num_top=20, by='num_citations'):
    if by == 'num_citations':
        return mysql.query(
            """SELECT * FROM top_keywords_by_num_citations LIMIT :num_top;""",
            {
                "num_top": num_top,
            },
        )
    elif by == 'num_publications':
        return mysql.query(
            """SELECT * FROM top_keywords_by_num_publications LIMIT :num_top;""",
            {
                "num_top": num_top,
            },
        )
    else:
        raise ValueError(f"{by=} not recognized.")


@app.callback(
    Output(component_id='figure_most_popular_keywords', component_property='figure'),
    Input(component_id='radio_by_most_popular_keywords', component_property='value'),
)
def make_figure_most_popular_keywords(by='num_citations'):
    df = get_most_popular_keywords(by=by)
    if by == 'num_citations':
        fig = px.bar(df, x="name", y="total_num_citations", height=300)
    elif by == 'num_publications':
        fig = px.bar(df, x="name", y="total_num_publications", height=300)
    return fig


def make_stores():
    return html.Div([
        dcc.Store(id='current_keyword'),
    ])


@app.callback(
    Output('current_keyword', 'data'),
    Input('figure_most_popular_keywords', 'clickData'),
)
def display_click_data(clickData):
    print(clickData)
    try:
        label = clickData['points'][0]['label']
    except TypeError:
        label = None
    return label


@app.callback(
    Output('current_keyword_text', 'children'),
    Input('current_keyword', 'data'),
)
def update_current_keyword_text(current_keyword):
    return f"{current_keyword=}"


def make_widget_most_popular_keywords(num_top=20, by='num_citations'):
    radio_by = dcc.RadioItems(
        id='radio_by_most_popular_keywords',
        options=['num_citations', 'num_publications'],
        value='num_citations',
        inline=False,
    )
    graph = dcc.Graph(id="figure_most_popular_keywords")
    widget = make_widget(
        title="Top keywords",
        subtitle="Keywords that have accumulated the most number of citations over all years",
        children=[radio_by, graph],
    )
    return widget


df_neo4j = neo4j.query("MATCH (f:FACULTY) RETURN f.name AS name, f.email AS email LIMIT 10")

df_mongodb = mongodb.query(
    lambda db: db.faculty.
    aggregate([{"$match": {"position": "Assistant Professor"}}, {"$project": {"_id": 0, "name": 1, "email": 1, "phone": 1}}])
)

colors = {
    'background': '#ffffff',
    'background1': '#73e8ff',
    'text': '#212121',
    'text1': '#000000',
}

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame(
    {
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"], "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    }
)

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text'],
)


def make_header():
    return dbc.Row(
        children=[
            dbc.Col(
                width=4,
                children=[
                    html.H1(
                        style={
                            # 'textAlign': 'center',
                            'color': colors['text'],
                        },
                        children='Research idea generator',
                    ),
                    html.Div(
                        style={
                            # 'textAlign': 'center',
                            'color': colors['text'],
                        },
                        children='Find your next research topic, collaborators, and references!',
                    ),
                ],
            ),
            dbc.Col(
                style={'border': '1px solid rgba(0, 0, 0, 0.2)'},
                width=8,
                children=[html.P(id="current_keyword_text")],
            )
        ]
    )


def make_widget(title="<title>", subtitle="<subtitle>", children=[]):
    return dbc.Col(
        style={
            'display': 'flex',
            'flexDirection': 'column',
            'overflow': 'hidden',
            'height': '450px',
            'padding': '16px',
        },
        width=4,
        children=[
            html.H4(title),
            html.P(subtitle),
            dbc.Card(
                style={
                    'overflow': 'auto',
                },
                children=children,
            ),
        ],
    )


placeholder_box = dbc.Col(
    width=6,
    children=["HELLO!"],
)


def make_widgets():
    return dbc.Row(
        style={
            'flex': '1 0 0',
            'overflow': 'auto',
        },
        children=[
            make_widget_most_popular_keywords(),
            make_widget(
                title="Hello, World!",
                children=[dcc.Graph(id='example-graph-2', figure=fig)],
            ),
            make_widget(children=[dash_table.DataTable(*df_to_dash_data_table(df_mongodb))]),
            make_widget(children=[dash_table.DataTable(*df_to_dash_data_table(df_mongodb))]),
            make_widget(children=[dash_table.DataTable(*df_to_dash_data_table(df_mongodb))]),
            make_widget(children=[dash_table.DataTable(*df_to_dash_data_table(df_mongodb))]),
        ]
    )


app.layout = dbc.Col(
    style={
        'display': 'flex',
        'flexDirection': 'column',
        'backgroundColor': colors['background1'],
        'fontFamily': 'sans-serif',
        'padding': '16px',
        'width': '100vw',
        'height': '100vh',
        'overflow': 'hidden',
    },
    children=[
        make_header(),
        make_widgets(),
        make_stores(),
    ]
)

if __name__ == '__main__':
    app.run_server(debug=True)
