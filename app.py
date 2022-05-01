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
    return html.Div(
        [
            dcc.Store(id='current_keyword'),
            dcc.Store(id='current_publication'),
            dcc.Store(id='current_researcher'),
        ]
    )


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


def query_researchers_for_keyword(current_keyword):
    res = mongodb.query(
        lambda db: db.faculty.aggregate(
            [
                {
                    "$match": {"keywords.name": current_keyword},
                },
                {
                    "$unwind": "$keywords",
                },
                {
                    "$match": {"keywords.name": current_keyword},
                },
                {
                    "$project": {"_id": 0},
                },
                {
                    "$sort": {"keywords.score": -1},
                },
                {
                    "$limit": 20,
                },
            ]
        )
    )

    if len(res) == 0:
        return None
    else:
        return res


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


def get_current_researcher_list_from_backend():
    res = mongodb.query(
        lambda db: db.researcher_list.
        aggregate([
            {
                "$project": {"_id": 0, "keywords": 0, "affiliation": 0, "publications": 0},
            },
        ])
    )

    if len(res) == 0:
        return None
    else:
        return res


def get_researcher_by_id(faculty_id):
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


@app.callback(
    Output('info_text', 'children'),
    [
        Input('current_keyword', 'data'),
        Input('current_publication', 'data'),
        Input('current_researcher', 'data'),
    ],
)
def update_info_text(current_keyword, current_publication, current_researcher):
    msg = '\n'.join([
        f"{current_keyword=}",
        f"{current_publication=}",
        f"{current_researcher=}",
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
    Output('current_researcher', 'data'),
    Input({'type': 'researcher_add_to_list_button', 'index': ALL}, 'n_clicks'),
)
def update_by_researcher_add_to_list_button(n_clicks):
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


@app.callback(
    Output('researcher_list', 'data'),
    [
        Input('current_researcher', 'data'),
        Input('button_delete_all_researchers', 'n_clicks'),
    ],
)
def update_researcher_list(current_researcher, n_clicks):
    print(f"{current_researcher=}")
    triggered = dash.callback_context.triggered[0]
    print(f"{triggered=}")

    if triggered['prop_id'] == 'button_delete_all_researchers.n_clicks':
        res = mongodb.query(lambda db: db.researcher_list.delete_many({}))
    elif current_researcher is not None and triggered is not None:
        researcher = get_researcher_by_id(current_researcher)
        print(researcher)
        res = mongodb.query(lambda db: db.researcher_list.insert_one(researcher))
        print(f"{res=}")

    researcher_list_data = get_current_researcher_list_from_backend()
    print(f"{researcher_list_data=}")

    return researcher_list_data


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


@app.callback(
    Output('top_researchers_widget', 'children'),
    Input('current_keyword', 'data'),
)
def update_top_researchers_widget(current_keyword):
    researchers = query_researchers_for_keyword(current_keyword)

    if researchers is None:
        return []
    else:
        children = []
        for r in researchers:
            print(f"{r=}")
            # r = {
            #     'id': 288,
            #     'name': 'Michael Rubenstein',
            #     'position': 'Assistant Professor of Computer Science',
            #     'researchInterest': 'The Lisa Wissner-Slivka and Benjamin Slivka Professor in Computer Science',
            #     'email': None,
            #     'phone': None,
            #     'affiliation':
            #         {
            #             'id': 6, 'name': 'Northwestern University',
            #             'photoUrl': 'https://www.northwestern.edu/brand/images/nu-horizontal.jpg'
            #         },
            #     'photoUrl': 'https://robotics.northwestern.edu/images/people/faculty/rubenstein-michael.jpg',
            #     'keywords': {'id': 199, 'name': 'algorithms', 'score': 13.6756},
            #     'publications':
            #         [
            #             69434499, 1517064069, 1563547082, 1572286800, 1651492067, 1878418498, 1982636603, 2023850386,
            #             2030775133, 2097063056, 2102560187, 2106675317, 2116735242, 2121104773, 2133846610, 2137941340,
            #             2139714790, 2143048619, 2153315991, 2154007976, 2167880800, 2167983353, 2473332233, 2563354324,
            #             2564629425, 2621081140, 2771764619, 2773163868, 2790555425, 2891489068, 2892256368, 2913084927,
            #             2915064205, 2934528086, 3000667810, 3001424831, 3003817311, 3008404222, 3038007248, 3043048731,
            #             3043258647, 3089977517
            #         ],
            # }
            id = r["id"]

            # keyword_tags = []
            # for k in r.get("keywords", []):
            #     keyword_name = k["name"]

            #     if keyword_name == current_keyword:
            #         tag_background = 'rgba(0, 0, 255, 0.2)'
            #     else:
            #         tag_background = 'rgba(0, 0, 0, 0.1)'

            #     tag = html.Div(
            #         style={
            #             'fontSize': '0.8em',
            #             'backgroundColor': tag_background,
            #             'color': 'rgba(0, 0, 0, 0.5)',
            #         },
            #         children=keyword_name,
            #     )
            #     keyword_tags.append(tag)

            children.append(
                dbc.Col(
                    width=1,
                    children=[
                        dbc.Card(
                            [
                                dbc.CardImg(
                                    style={
                                        "objectFit": "cover",
                                        "height": "160px",
                                    },
                                    src=r['photoUrl'],
                                    top=True,
                                ),
                                dbc.CardBody(
                                    [
                                        html.H4(r["name"], className="card-title"),
                                        html.Div(f"{r['position']}", className="card-text")
                                        if r['position'] is not None else None,
                                        html.Div(
                                            style={
                                                'fontSize': '0.8em',
                                                'color': 'rgba(0, 0, 0, 0.5)',
                                            },
                                            children=[
                                                html.Div(f"{r['affiliation']['name']}"),
                                                html.Div(f"{r['email']}") if r['email'] is not None else None,
                                                html.Div(f"Score {r['keywords']['score']}"),
                                            ]
                                        ),
                                        dbc.Button(
                                            id={
                                                'type': 'researcher_add_to_list_button',
                                                'index': id,
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
            make_widget(
                title="Your reference publications list",
                badges=["MongoDB", "backend-updating"],
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
                subtitle="The researchers that have the highest score on the keyword.",
                width=12,
                height=None,
                children=[html.Div(id="top_researchers_widget")],
            ),
            make_widget(
                title="Your researchers list",
                badges=["MongoDB", "backend-updating"],
                subtitle="Add or potential collaborators you might want to work with",
                width=12,
                height=None,
                children=[
                    dbc.Button(
                        id="button_delete_all_researchers",
                        color='primary',
                        children='Delete all researchers',
                    ),
                    dash_table.DataTable(
                        id="researcher_list",
                        columns=[
                            {"name": "id", "id": "id"},
                            {"name": "name", "id": "name"},
                            {"name": "position", "id": "position"},
                        ],
                    ),
                ],
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
