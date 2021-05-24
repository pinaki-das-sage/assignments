import os
from pathlib import Path

import pandas as pd
from flask import render_template
import plotly.express as px
import plotly
import json


class Assignment5:
    movies = None

    def __init__(self):
        filename = os.path.join(Path(__file__).parent, 'data', '5_imdb_top_1000.csv')
        self.movies = pd.read_csv(filename)

    def process(self):
        # create a earnings column from gross by replacing all ,
        self.movies['Earnings'] = self.movies['Gross'].str.replace(',', '')
        movies = self.movies.astype({'Earnings': float})

        # create a new column for year
        movies['Year'] = movies['Released_Year']

        # there's a stray PG value in the Year column, filter it out
        movies['Year'] = movies[movies['Year'] != 'PG']['Year']

        # drop null values from Year column
        movies['Year'].dropna(inplace=True)

        # group by year but retain it as a column (dont make it an index)
        groupedMoviesList = movies.groupby('Year', as_index=False)

        # get a average of the ratings per year
        averageRatingByYear = groupedMoviesList.mean()

        # create a line chart out of it
        fig = px.line(
            averageRatingByYear,
            x="Year",
            y="IMDB_Rating",
            title='Average movie rating by year (hover to see average earnings)',
            hover_data=["Earnings"])

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        datasource = "https://www.kaggle.com/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows"
        return render_template("assignment5.html.j2", graphJSON=graphJSON, datasource=datasource)
