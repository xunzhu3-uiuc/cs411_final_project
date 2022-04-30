Find your next research topic!
==============================

Purpose
-------

This application is my fellow researchers who is struggling to find their next
research topic. It gives you lots of tools to zero in just exactly what you want
to work on for your next project.

Do you want to focus on a "hot" topic that might give you lots of citations?
Do you have a particular collaborator that you want to work with?
Or perhaps you want to write an review on a particular topic, but need to first gather all the recent papers?

You will find tools to answer these questions in this nice application!

Demo
----

TODO: LINK


Installation
------------

First, install Dash:

```
$ pip install dash
```

Then, clone this repo:

```
$ git clone https://github.com/xunzhu3-uiuc/cs411_final_project.git
$ cd cs411_final_project
```

Next, run the Dash application:

```
$ python app.py
```

Finally, open your favorite browser and go to

```
http://localhost:8050
```

and start your exploration!


Usage
-----

All widgets are point-and-click and are pretty self-explanatory. If you need
help, contact me at xunzhu3@illinois.edu. ;)


Design
------

This is a very typical Dash app that uses [Dash Bootstrap
Components](https://dash-bootstrap-components.opensource.faculty.ai/) for the
baseline CSS.

The communication with MySQL, MongoDB, and Neo4J are through their official
Python Drivers. The returned tables are usually converted into Pandas Dataframes
for final modifications.

Implementation
--------------

The web app is entirely written in Python using Dash.

The view creation is done using the original query language based on the database.

Database Techniques
-------------------

  - **Indexing.** Slow queries are optimized using indexing.
  - **View.** Many complex queries are first converted into views before queried.
  - **Prepared statements**. All queries with variable parameters are done using prepared statements.
  - **Transaction**. All multi-statement queries are contained in transactions.
<!-- REST API for accessing databases -->
<!-- Constraint -->
<!-- Trigger -->
<!-- Stored procedure -->
<!-- Parallel query execution -->
<!-- Partitioning/sharding -->

Extra-Credit Capabilities
-------------------------

  - **Multi-database querying**
  - **Data expansion**

Contributions
-------------

This is a one-person effort done my me (Xun Zhu xunzhu3@illinois.edu). The
entire project took me about 25 hours to finish.
