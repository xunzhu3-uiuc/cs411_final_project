# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, dcc, html, dash_table, Input, Output, State, MATCH, ALL
import json
import dash
import plotly.express as px
import pandas as pd
from sqlalchemy import create_engine, text
from pymongo import MongoClient
from neo4j import GraphDatabase
from flask_caching import Cache
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto

CACHE_TIMEOUT_SECONDS = 300000

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


class MongoDB:

    def __init__(self):
        conn = MongoClient('mongodb://xzhu:t817@localhost:27017/academicworld?authSource=admin')
        self._db = conn['academicworld']

    def query(self, query_fn):
        cursor = query_fn(self._db)
        try:
            res = list(cursor)
        except Exception:
            res = cursor
        return res


class MySQL:

    def __init__(self):
        self._engine = create_engine("mysql+pymysql://xzhu:t817@localhost/AcademicWorld", echo=True, future=True)

    def query(self, query_str, params=[]):
        conn = self._engine.connect()
        result = pd.read_sql(text(query_str), conn, params=params)
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


def make_stores():
    return html.Div([
        dcc.Store(id='current_keyword'),
        dcc.Store(id='current_publication'),
    ])


def query_related_keywords(current_keyword):
    nodes, relationships = neo4j.query_graph(
        f'MATCH p = (:KEYWORD {{name: "{current_keyword}"}})-[:LABEL_BY*..2]-(:KEYWORD) RETURN p LIMIT 15'
    )

    # print(f"{current_keyword=}")
    vertices = []
    for n in nodes:
        node_id = n.get("id")
        node_label = n.get("name") if ("KEYWORD" in n.labels) else n.get("title")
        # print(f"{node_label=}")
        # print(f"{(node_label == current_keyword)=}")
        if node_label == current_keyword:
            node_color = "lightgreen"
        elif "KEYWORD" in n.labels:
            node_color = "blue"
        else:
            node_color = "red"
        vertex = {
            "data": {
                "id": node_id,
                "label": node_label,
            },
            "style": {"background-color": node_color},
        }
        vertices.append(vertex)

    # print(f"{vertices=}")

    edges = [{"data": {
        "source": r.start_node.get("id"),
        "target": r.end_node.get("id"),
    }} for r in relationships]

    return vertices + edges


def query_publications_for_keyword(current_keyword):
    res = mongodb.query(
        lambda db: db.publications.aggregate(
            [
                {
                    "$match": {"keywords.name": current_keyword},
                },
                {
                    "$project": {"_id": 0},
                },
                {"$sort": {"numCitations": -1}},
                {"$limit": 20},
            ]
        )
    )

    if len(res) == 0:
        return None
    else:
        return res
        # df = pd.DataFrame(res)
        # df['keywords'] = df['keywords'].apply(lambda x: str(x))
        # return df


def query_researchers_for_keyword(current_keyword):
    res = mongodb.query(
        lambda db: db.faculty.aggregate(
            [
                {
                    "$match": {"keywords.name": current_keyword},
                },
                {
                    "$project": {"_id": 0},
                },
                # {"$sort": {"numCitations": -1}},
                {"$limit": 20},
            ]
        )
    )

    if len(res) == 0:
        return None
    else:
        df = pd.DataFrame(res)
        df['affiliation'] = df['affiliation'].apply(lambda x: str(x))
        df['publications'] = df['publications'].apply(lambda x: str(x))
        df['keywords'] = df['keywords'].apply(lambda x: str(x))
        return df


def get_publication_by_id(publication_id):
    res = mongodb.query(
        lambda db: db.publications.aggregate([
            {
                "$match": {"id": publication_id},
            },
            {
                "$project": {"_id": 0},
            },
            {"$limit": 1},
        ])
    )

    if len(res) == 0:
        return None
    else:
        return res[0]


def get_current_publication_list_from_backend():
    res = mongodb.query(
        lambda db: db.publication_list.aggregate(
            [
                # {
                #     "$match": {"id": publication_id},
                # },
                {
                    "$project": {"_id": 0, "keywords": 0},
                },
                # {"$limit": 1},
            ]
        )
    )

    if len(res) == 0:
        return None
    else:
        return res


def get_faculty_by_id(faculty_id):
    res = mongodb.query(
        lambda db: db.faculty.aggregate([
            {
                "$match": {"id": faculty_id},
            },
            {
                "$project": {"_id": 0},
            },
            {"$limit": 1},
        ])
    )

    if len(res) == 0:
        return None
    else:
        return res[0]


# @app.callback(
#     [
#         Output('publications_for_keyword', 'data'),
#         Output('publications_for_keyword', 'columns'),
#     ],
#     Input('current_keyword', 'data'),
# )
# def update_publications_for_keyword(current_keyword):
#     df = query_publications_for_keyword(current_keyword)

#     if df is not None:
#         outputs = df_to_dash_data_table(df)
#         return outputs
#     else:
#         return [[], []]


@app.callback(
    [
        Output('researchers_for_keyword', 'data'),
        Output('researchers_for_keyword', 'columns'),
    ],
    Input('current_keyword', 'data'),
)
def update_researchers_for_keyword(current_keyword):
    df = query_researchers_for_keyword(current_keyword)

    if df is not None:
        outputs = df_to_dash_data_table(df)
        return outputs
    else:
        return [[], []]


@app.callback(
    Output('figure_most_popular_keywords', 'figure'),
    Input('radio_by_most_popular_keywords', 'value'),
)
def make_figure_most_popular_keywords(by='num_citations'):
    df = query_most_popular_keywords(by=by)
    if by == 'num_citations':
        fig = px.bar(df, x="name", y="total_num_citations", height=280)
    elif by == 'num_publications':
        fig = px.bar(df, x="name", y="total_num_publications", height=280)
    return fig


@app.callback(
    Output('current_keyword', 'data'),
    [
        Input('figure_most_popular_keywords', 'clickData'),
        Input('related_keywords', 'tapNodeData'),
    ],
)
def display_click_data(data1, data2):
    ctx = dash.callback_context

    for event in ctx.triggered:
        prop_id = event['prop_id']
        value = event['value']

        if prop_id == 'related_keywords.tapNodeData':
            try:
                label = value['label']
            except TypeError:
                label = None
        elif prop_id == 'figure_most_popular_keywords.clickData':
            try:
                label = value['points'][0]['label']
            except TypeError:
                label = None
        else:
            label = None

    return label or "algorithms"


@app.callback(Output('info_text', 'children'), [
    Input('current_keyword', 'data'),
    Input('current_publication', 'data'),
])
def update_info_text(current_keyword, current_publication):
    msg = '\n'.join([
        f"{current_keyword=}",
        f"{current_publication=}",
    ])
    return msg


@app.callback(
    Output('related_keywords', 'elements'),
    Input('current_keyword', 'data'),
)
def update_related_keywords(current_keyword):
    elements = query_related_keywords(current_keyword)
    return elements


@app.callback(
    Output('current_publication', 'data'),
    Input({'type': 'publication_add_to_list_button', 'index': ALL}, 'n_clicks'),
)
def update_by_publication_add_to_list_button(n_clicks):
    try:
        triggered = dash.callback_context.triggered[0]
        id_str = json.loads(triggered['prop_id'].split(".")[0])["index"]
    except Exception:
        id_str = None
    return id_str


@app.callback(
    Output('publication_list', 'data'),
    [
        Input('current_publication', 'data'),
        Input('button_delete_all_publications', 'n_clicks'),
    ],
)
def update_publication_list(current_publication, n_clicks):
    print(f"{current_publication=}")
    triggered = dash.callback_context.triggered[0]
    print(f"{triggered=}")

    if triggered['prop_id'] == 'button_delete_all_publications.n_clicks':
        res = mongodb.query(lambda db: db.publication_list.delete_many({}))
    elif current_publication is not None and triggered is not None:
        publication = get_publication_by_id(current_publication)
        print(publication)
        res = mongodb.query(lambda db: db.publication_list.insert_one(publication))
        print(f"{res=}")

    publication_list_data = get_current_publication_list_from_backend()
    print(f"{publication_list_data=}")

    return publication_list_data


# @app.callback(
#     Output('selected_publication', 'data'),
#     Input('publications_for_keyword', 'active_cell'),
# )
# def update_selected_publication(active_cell):
#     print(active_cell)
#     return None


@app.callback(
    Output('top_publications_widget', 'children'),
    Input('current_keyword', 'data'),
)
def update_top_publications_widget(current_keyword):
    publications = query_publications_for_keyword(current_keyword)

    if publications is None:
        return []
    else:
        children = []
        for pub in publications:
            # pub = {
            #     'id': 2109184569,
            #     'title': 'PHENIX: building new software for automated crystallographic structure determination',
            #     'venue': 'Acta Crystallographica Section D-biological Crystallography',
            #     'year': 2002,
            #     'numCitations': 4227,
            #     'keywords':
            #         [
            #             {'id': 199, 'name': 'algorithms', 'score': 0.00218301},
            #             {'id': 1198, 'name': 'databases', 'score': 0.00861767},
            #             {'id': 2841, 'name': 'challenges', 'score': 0.00203716},
            #             {'id': 2871, 'name': 'python', 'score': 0.186769},
            #             {'id': 6354, 'name': 'genomics', 'score': 0.347225},
            #             {'id': 8370, 'name': 'protein', 'score': 0.0038914},
            #             {'id': 11437, 'name': 'expert', 'score': 0.0861387},
            #             {'id': 19259, 'name': 'projects', 'score': 0.00321668},
            #             {'id': 28214, 'name': 'software package', 'score': 0.226737},
            #         ],
            # }
            pub_id = pub["id"]

            if pub['venue'] is None:
                venue_div = None
            else:
                venue_div = html.Div(f"{pub['venue']}", className="card-text")

            keyword_tags = []
            for k in pub.get("keywords", []):
                keyword_name = k["name"]

                if keyword_name == current_keyword:
                    tag_background = 'rgba(0, 0, 255, 0.2)'
                else:
                    tag_background = 'rgba(0, 0, 0, 0.1)'

                tag = html.Div(
                    style={
                        'fontSize': '0.8em',
                        'backgroundColor': tag_background,
                        'color': 'rgba(0, 0, 0, 0.5)',
                    },
                    children=keyword_name,
                )
                keyword_tags.append(tag)

            children.append(
                dbc.Col(
                    width=3,
                    children=[
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H4(pub["title"], className="card-title"),
                                        venue_div,
                                        html.Div(
                                            style={
                                                'fontSize': '0.8em',
                                                'color': 'rgba(0, 0, 0, 0.5)',
                                            },
                                            className="card-text",
                                            children=f"Published in year {pub['year']} | Cited {pub['numCitations']} times",
                                        ),
                                        html.Div(
                                            style={
                                                'width': '100%',
                                                'marginBottom': '8px',
                                                'display': 'flex',
                                                'flexDirection': 'row',
                                                'flexWrap': 'wrap',
                                                'gap': '4px',
                                            },
                                            children=keyword_tags,
                                        ),
                                        dbc.Button(
                                            id={
                                                'type': 'publication_add_to_list_button',
                                                'index': pub_id,
                                            },
                                            color='primary',
                                            children='Add to list',
                                        ),
                                    ],
                                ),
                            ]
                        )
                    ]
                )
            )
        return html.Div(
            style={
                'width': '100%',
                'display': 'flex',
                'flexDirection': 'row',
                'margin': '16px',
                'gap': '16px',
            },
            children=children,
        )


colors = {
    'background': '#ffffff',
    'background1': '#73e8ff',
    'text': '#212121',
    'text1': '#000000',
}


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
                children=[html.Pre(id='info_text')],
            )
        ]
    )


def make_widget(title="<title>", subtitle="<subtitle>", badges=[], children=[], width=4, height='450px'):
    return dbc.Col(
        style={
            'display': 'flex',
            'flexDirection': 'column',
            'overflow': 'hidden',
            'height': height,
            'padding': '16px',
        },
        width=width,
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


def make_widgets():
    return dbc.Row(
        style={
            'flex': '1 0 0',
            'overflow': 'auto',
        },
        children=[
            make_widget(
                title="Top keywords",
                badges=["MySQL", "input from user", "indexing", "views", "cached results", "dash stores", "click data"],
                subtitle="Keywords that have accumulated the most number of citations over all years",
                width=6,
                children=[
                    html.Div(
                        style={
                            "display": 'flex',
                            'flexDirection': 'row',
                        },
                        children=[
                            "Based on: ",
                            dcc.RadioItems(
                                id='radio_by_most_popular_keywords',
                                options=['num_citations', 'num_publications'],
                                value='num_citations',
                                inline=True,
                            ),
                        ]
                    ),
                    dcc.Graph(id="figure_most_popular_keywords")
                ],
            ),
            make_widget(
                title="Related keywords",
                badges=["Neo4J", "Cytoscape", "input from user"],
                subtitle="Click a keyword on the first panel, and see what are the common related keywords.",
                width=6,
                children=[
                    cyto.Cytoscape(
                        id='related_keywords',
                        style={'width': '100%', 'height': '400px'},
                        elements=[],
                        layout={'name': 'cose'},
                    ),
                ],
            ),
            make_widget(
                title="Top publications",
                badges=["MongoDB", "pattern-matching callbacks"],
                subtitle="The most-cited publications given the keyword.",
                width=12,
                height=None,
                children=[html.Div(id="top_publications_widget")],
            ),
            # make_widget(
            #     title="Top publications",
            #     badges=["MongoDB"],
            #     subtitle="The most-cited publications given the keyword.",
            #     width=12,
            #     children=[dash_table.DataTable(id="publications_for_keyword")],
            # ),
            make_widget(
                title="Your reference publications list",
                badges=["MongoDB", "MySQL", "backend-updating"],
                subtitle="Add or important papers you should read based on your selections.",
                width=12,
                height=None,
                children=[
                    dbc.Button(
                        id="button_delete_all_publications",
                        color='primary',
                        children='Delete all publications',
                    ),
                    dash_table.DataTable(
                        id="publication_list", columns=[{"name": "id", "id": "id"}, {"name": "title", "id": "title"}]
                    ),
                ],
            ),
            make_widget(
                title="Top researchers",
                badges=["MongoDB", "pattern-matching callbacks"],
                subtitle="The most-cited researchers given the keyword.",
                width=12,
                children=[dash_table.DataTable(id="researchers_for_keyword")],
            ),
            make_widget(
                title="Your next collaborators",
                badges=["MongoDB", "MySQL", "backend-updating"],
                subtitle="Add or delete your researchers you want to work with.",
                width=12,
                height=None,
                children=[],
            ),
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
