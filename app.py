# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, dcc, html, dash_table
import plotly.express as px
import pandas as pd
from sqlalchemy import create_engine, text
from pymongo import MongoClient
from neo4j import GraphDatabase


class Neo4J:

    def __init__(self):
        driver = GraphDatabase.driver("bolt://localhost:7687/academicworld", auth=("neo4j", "t817"))
        session = driver.session(database="academicworld")
        self._session = session

    def query(self, query_str):
        result = self._session.run(query_str)
        df = pd.DataFrame.from_records(result.data())
        return df


neo4j = Neo4J()
df_neo4j = neo4j.query("MATCH (f:FACULTY) RETURN f.name AS name, f.email AS email LIMIT 10")


class MongoDB:

    def __init__(self):
        conn = MongoClient('mongodb://xzhu:t817@localhost:27017/academicworld?authSource=admin')
        self._db = conn['academicworld']

    def query(self, query_fn):
        cursor = query_fn(self._db)
        df = pd.DataFrame(list(cursor))
        return df


mongodb = MongoDB()
df_mongodb = mongodb.query(
    lambda db: db.faculty.
    aggregate([{"$match": {"position": "Assistant Professor"}}, {"$project": {"_id": 0, "name": 1, "email": 1, "phone": 1}}])
)


class MySQL:

    def __init__(self):
        engine = create_engine("mysql+pymysql://xzhu:t817@localhost/AcademicWorld", echo=True, future=True)
        conn = engine.connect()
        self._conn = conn

    def query(self, query_str):
        result = pd.read_sql(text(query_str), self._conn)
        return result


mysql = MySQL()
df_mysql = mysql.query("SELECT * FROM faculty WHERE position = 'Assistant Professor'")


def df_to_dash_data_table(df):
    return [df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]]


app = Dash(__name__)

colors = {'background': '#111111', 'text': '#7FDBFF'}

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame(
    {
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"], "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    }
)

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

fig.update_layout(plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], font_color=colors['text'])

app.layout = html.Div(
    style={'backgroundColor': colors['background']},
    children=[
        html.H1(children='Hello Dash', style={'textAlign': 'center', 'color': colors['text']}),
        html.Div(
            children='Dash: A web application framework for your data.', style={'textAlign': 'center', 'color': colors['text']}
        ),
        dcc.Graph(id='example-graph-2', figure=fig),
        dash_table.DataTable(*df_to_dash_data_table(df_mongodb)),
        # dash_table.DataTable(result.to_dict('records'), [{"name": i, "id": i} for i in result.columns])
    ]
)

if __name__ == '__main__':
    app.run_server(debug=True)
