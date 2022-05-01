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
import dash_cytoscape as cyto

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
        self._driver = driver

    def query_graph(self, query_str):
        session = self._driver.session(database="academicworld")
        result = session.run(query_str)
        graph = result.graph()

        return graph.nodes, graph.relationships


# [<Node id=2960 labels=frozenset({'KEYWORD'}) properties={'name': 'algorithm', 'id': 'k189'}>
#  <Node id=524095 labels=frozenset({'PUBLICATION'}) properties={'venue': 'arXiv preprint arXiv:1011.2121', 'year': 2010, 'numCitations': 0, 'id': 'p2952868510', 'title': 'Matching with Couples Revisited'}>
#  <Node id=13823 labels=frozenset({'KEYWORD'}) properties={'name': 'game', 'id': 'k2586'}>
#  <Node id=643 labels=frozenset({'PUBLICATION'}) properties={'venue': 'Proceedings of the 2016 Annual Symposium on Computer-Human Interaction in Play Companion Extended Abstracts', 'year': 2016, 'numCitations': 0, 'id': 'p2530829023', 'title': 'Treehouse Dreams: A Game-Based Method for Eliciting Interview Data from Children'}>
#  <Node id=572959 labels=frozenset({'PUBLICATION'}) properties={'venue': 'nan', 'year': 2020, 'numCitations': 0, 'id': 'p3109138948', 'title': 'METHOD AND APPARATUS FOR REPLACING DATA FROM NEAR TO FAR MEMORY OVER A SLOW INTERCONNECT FOR OVERSUBSCRIBED IRREGULAR APPLICATIONS'}>
#  <Node id=12097 labels=frozenset({'KEYWORD'}) properties={'name': 'memories', 'id': 'k49485'}>
#  <Node id=644 labels=frozenset({'PUBLICATION'}) properties={'venue': 'Proceedings of the 14th ACM/IEEE Symposium on Embedded Systems for Real-Time Multimedia', 'year': 2016, 'numCitations': 3, 'id': 'p2530832681', 'title': 'On Detecting and Using Memory Phases in Multimedia Systems'}>
#  <Node id=151253 labels=frozenset({'PUBLICATION'}) properties={'venue': 'Neuroinformatics', 'year': 2011, 'numCitations': 121, 'id': 'p2157408013', 'title': 'Automated Reconstruction of Neuronal Morphology Based on Local Geometrical and Global Structural Models'}>
#  <Node id=69492 labels=frozenset({'KEYWORD'}) properties={'name': 'structural models', 'id': 'k52826'}>
#  <Node id=645 labels=frozenset({'PUBLICATION'}) properties={'venue': 'Journal of Abnormal Psychology', 'year': 2016, 'numCitations': 17, 'id': 'p2530843363', 'title': 'A comparison and integration of structural models of depression and anxiety in a clinical sample: Support for and validation of the tri-level model'}>
#  <Node id=496938 labels=frozenset({'PUBLICATION'}) properties={'venue': 'MBIA/MFCA@MICCAI', 'year': 2019, 'numCitations': 0, 'id': 'p2979897994', 'title': 'Species-Preserved Structural Connections Revealed by Sparse Tensor CCA'}>
#  <Node id=76019 labels=frozenset({'KEYWORD'}) properties={'name': 'canonical correlation analysis', 'id': 'k1628'}>
#  <Node id=646 labels=frozenset({'PUBLICATION'}) properties={'venue': 'arXiv preprint arXiv:1610.03454', 'year': 2016, 'numCitations': 77, 'id': 'p2530846021', 'title': 'Deep Variational Canonical Correlation Analysis'}>
#  <Node id=431587 labels=frozenset({'PUBLICATION'}) properties={'venue': 'IGARSS 2018 - 2018 IEEE International Geoscience and Remote Sensing Symposium', 'year': 2018, 'numCitations': 0, 'id': 'p2901282489', 'title': 'Deterministic Cramer-Rao Bound for Scanning Radar Sensing'}>
#  <Node id=15944 labels=frozenset({'KEYWORD'}) properties={'name': 'target', 'id': 'k24731'}>
#  <Node id=647 labels=frozenset({'PUBLICATION'}) properties={'venue': 'nan', 'year': 2014, 'numCitations': 22, 'id': 'p2530848804', 'title': 'Methods, systems, and media for authenticating users using multiple services'}>
# ]

# [<Relationship id=4410459 nodes=(
#     <Node id=524095 labels=frozenset({'PUBLICATION'}) properties={'venue': 'arXiv preprint arXiv:1011.2121', 'year': 2010, 'numCitations': 0, 'id': 'p2952868510', 'title': 'Matching with Couples Revisited'}>
#     <Node id=2960 labels=frozenset({'KEYWORD'}) properties={'name': 'algorithm', 'id': 'k189'}>)
#  type='LABEL_BY'
#  properties={'score': 0.00111362}>
#  <Relationship id=4410460 nodes=(<Node id=524095 labels=frozenset({'PUBLICATION'}) properties={'venue': 'arXiv preprint arXiv:1011.2121', 'year': 2010, 'numCitations': 0, 'id': 'p2952868510', 'title': 'Matching with Couples Revisited'}>
#  <Node id=13823 labels=frozenset({'KEYWORD'}) properties={'name': 'game', 'id': 'k2586'}>) type='LABEL_BY' properties={'score': 0.0015675}>
#  <Relationship id=3887232 nodes=(<Node id=643 labels=frozenset({'PUBLICATION'}) properties={'venue': 'Proceedings of the 2016 Annual Symposium on Computer-Human Interaction in Play Companion Extended Abstracts', 'year': 2016, 'numCitations': 0, 'id': 'p2530829023', 'title': 'Treehouse Dreams: A Game-Based Method for Eliciting Interview Data from Children'}>
#  <Node id=13823 labels=frozenset({'KEYWORD'}) properties={'name': 'game', 'id': 'k2586'}>) type='LABEL_BY' properties={'score': 0.00286448}>
#  <Relationship id=4450194 nodes=(<Node id=572959 labels=frozenset({'PUBLICATION'}) properties={'venue': 'nan', 'year': 2020, 'numCitations': 0, 'id': 'p3109138948', 'title': 'METHOD AND APPARATUS FOR REPLACING DATA FROM NEAR TO FAR MEMORY OVER A SLOW INTERCONNECT FOR OVERSUBSCRIBED IRREGULAR APPLICATIONS'}>
#  <Node id=2960 labels=frozenset({'KEYWORD'}) properties={'name': 'algorithm', 'id': 'k189'}>) type='LABEL_BY' properties={'score': 0.00240893}>
#  <Relationship id=4450200 nodes=(<Node id=572959 labels=frozenset({'PUBLICATION'}) properties={'venue': 'nan', 'year': 2020, 'numCitations': 0, 'id': 'p3109138948', 'title': 'METHOD AND APPARATUS FOR REPLACING DATA FROM NEAR TO FAR MEMORY OVER A SLOW INTERCONNECT FOR OVERSUBSCRIBED IRREGULAR APPLICATIONS'}>
#  <Node id=12097 labels=frozenset({'KEYWORD'}) properties={'name': 'memories', 'id': 'k49485'}>) type='LABEL_BY' properties={'score': 0.315763}>
#  <Relationship id=3887243 nodes=(<Node id=644 labels=frozenset({'PUBLICATION'}) properties={'venue': 'Proceedings of the 14th ACM/IEEE Symposium on Embedded Systems for Real-Time Multimedia', 'year': 2016, 'numCitations': 3, 'id': 'p2530832681', 'title': 'On Detecting and Using Memory Phases in Multimedia Systems'}>
#  <Node id=12097 labels=frozenset({'KEYWORD'}) properties={'name': 'memories', 'id': 'k49485'}>) type='LABEL_BY' properties={'score': 0.297633}>
#  <Relationship id=3773791 nodes=(<Node id=151253 labels=frozenset({'PUBLICATION'}) properties={'venue': 'Neuroinformatics', 'year': 2011, 'numCitations': 121, 'id': 'p2157408013', 'title': 'Automated Reconstruction of Neuronal Morphology Based on Local Geometrical and Global Structural Models'}>
#  <Node id=2960 labels=frozenset({'KEYWORD'}) properties={'name': 'algorithm', 'id': 'k189'}>) type='LABEL_BY' properties={'score': 0.000785571}>
#  <Relationship id=3773799 nodes=(<Node id=151253 labels=frozenset({'PUBLICATION'}) properties={'venue': 'Neuroinformatics', 'year': 2011, 'numCitations': 121, 'id': 'p2157408013', 'title': 'Automated Reconstruction of Neuronal Morphology Based on Local Geometrical and Global Structural Models'}>
#  <Node id=69492 labels=frozenset({'KEYWORD'}) properties={'name': 'structural models', 'id': 'k52826'}>) type='LABEL_BY' properties={'score': 0.184492}>
#  <Relationship id=3887253 nodes=(<Node id=645 labels=frozenset({'PUBLICATION'}) properties={'venue': 'Journal of Abnormal Psychology', 'year': 2016, 'numCitations': 17, 'id': 'p2530843363', 'title': 'A comparison and integration of structural models of depression and anxiety in a clinical sample: Support for and validation of the tri-level model'}>
#  <Node id=69492 labels=frozenset({'KEYWORD'}) properties={'name': 'structural models', 'id': 'k52826'}>) type='LABEL_BY' properties={'score': 0.305557}>
#  <Relationship id=4462941 nodes=(<Node id=496938 labels=frozenset({'PUBLICATION'}) properties={'venue': 'MBIA/MFCA@MICCAI', 'year': 2019, 'numCitations': 0, 'id': 'p2979897994', 'title': 'Species-Preserved Structural Connections Revealed by Sparse Tensor CCA'}>
#  <Node id=2960 labels=frozenset({'KEYWORD'}) properties={'name': 'algorithm', 'id': 'k189'}>) type='LABEL_BY' properties={'score': 0.000334525}>
#  <Relationship id=4462943 nodes=(<Node id=496938 labels=frozenset({'PUBLICATION'}) properties={'venue': 'MBIA/MFCA@MICCAI', 'year': 2019, 'numCitations': 0, 'id': 'p2979897994', 'title': 'Species-Preserved Structural Connections Revealed by Sparse Tensor CCA'}>
#  <Node id=76019 labels=frozenset({'KEYWORD'}) properties={'name': 'canonical correlation analysis', 'id': 'k1628'}>) type='LABEL_BY' properties={'score': 0.100949}>
#  <Relationship id=3887256 nodes=(<Node id=646 labels=frozenset({'PUBLICATION'}) properties={'venue': 'arXiv preprint arXiv:1610.03454', 'year': 2016, 'numCitations': 77, 'id': 'p2530846021', 'title': 'Deep Variational Canonical Correlation Analysis'}>
#  <Node id=76019 labels=frozenset({'KEYWORD'}) properties={'name': 'canonical correlation analysis', 'id': 'k1628'}>) type='LABEL_BY' properties={'score': 0.352635}>
#  <Relationship id=4547998 nodes=(<Node id=431587 labels=frozenset({'PUBLICATION'}) properties={'venue': 'IGARSS 2018 - 2018 IEEE International Geoscience and Remote Sensing Symposium', 'year': 2018, 'numCitations': 0, 'id': 'p2901282489', 'title': 'Deterministic Cramer-Rao Bound for Scanning Radar Sensing'}>
#  <Node id=2960 labels=frozenset({'KEYWORD'}) properties={'name': 'algorithm', 'id': 'k189'}>) type='LABEL_BY' properties={'score': 0.00171165}>
#  <Relationship id=4548963 nodes=(<Node id=431587 labels=frozenset({'PUBLICATION'}) properties={'venue': 'IGARSS 2018 - 2018 IEEE International Geoscience and Remote Sensing Symposium', 'year': 2018, 'numCitations': 0, 'id': 'p2901282489', 'title': 'Deterministic Cramer-Rao Bound for Scanning Radar Sensing'}>
#  <Node id=15944 labels=frozenset({'KEYWORD'}) properties={'name': 'target', 'id': 'k24731'}>) type='LABEL_BY' properties={'score': 0.00234256}>
#  <Relationship id=3887268 nodes=(<Node id=647 labels=frozenset({'PUBLICATION'}) properties={'venue': 'nan', 'year': 2014, 'numCitations': 22, 'id': 'p2530848804', 'title': 'Methods, systems, and media for authenticating users using multiple services'}>
#  <Node id=15944 labels=frozenset({'KEYWORD'}) properties={'name': 'target', 'id': 'k24731'}>) type='LABEL_BY' properties={'score': 0.00247541}>]


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
def query_most_popular_keywords(num_top=20, by='num_citations'):
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
    df = query_most_popular_keywords(by=by)
    print(df)
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
        badges=["MySQL", "views", "cached results", "dash stores", "click data"],
        subtitle="Keywords that have accumulated the most number of citations over all years",
        children=[radio_by, graph],
    )
    return widget


# @cache.memoize(timeout=CACHE_TIMEOUT_SECONDS)
def query_related_keywords(current_keyword):
    nodes, relationships = neo4j.query_graph(
        f'MATCH p = shortestPath((:KEYWORD {{name: "{current_keyword}"}})-[:LABEL_BY*]-(f)) RETURN p LIMIT 5'
    )

    vertices = [
        {
            "data": {
                "id": n.get("id"),
                "label": n.get("name") if ("KEYWORD" in n.labels) else n.get("title"),
            },
            "style": {"background-color": "red" if ("KEYWORD" in n.labels) else "blue"}
        } for n in nodes
    ]

    edges = [{"data": {
        "source": r.start_node.get("id"),
        "target": r.end_node.get("id"),
    }} for r in relationships]

    return vertices + edges


def make_widget_related_keywords():
    # radio_by = dcc.RadioItems(
    #     id='radio_by_most_popular_keywords',
    #     options=['num_citations', 'num_publications'],
    #     value='num_citations',
    #     inline=False,
    # )
    # graph = dcc.Graph(id="figure_most_popular_keywords")
    widget = make_widget(
        title="Related keywords",
        badges=["Neo4J", "Cytoscape"],
        subtitle="Click a keyword on the first panel, and see what are the common related keywords.",
        children=[
            cyto.Cytoscape(
                id='related_keywords',
                style={'width': '100%', 'height': '400px'},
                elements=[
                    {'data': {'id': 'ca', 'label': 'Canada'}}, {'data': {'id': 'on', 'label': 'Ontario'}},
                    {'data': {'id': 'qc', 'label': 'Quebec'}}, {'data': {'source': 'ca', 'target': 'on'}},
                    {'data': {'source': 'ca', 'target': 'qc'}}
                ],
                layout={'name': 'cose'},
            ),
        ],
    )
    return widget


@app.callback(
    Output(component_id='related_keywords', component_property='elements'),
    Input(component_id='current_keyword', component_property='data'),
)
def update_related_keywords(current_keyword):
    elements = query_related_keywords(current_keyword)
    print(elements)
    return elements


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


def make_widget(title="<title>", subtitle="<subtitle>", badges=[], children=[]):
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
            html.Div(
                style={
                    'display': 'flex',
                    'flexDirection': 'row',
                    'gap': '4px',
                },
                children=[dbc.Badge(b) for b in badges],
            ),
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
            make_widget_related_keywords(),
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
