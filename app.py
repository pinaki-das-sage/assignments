import os

from flask import Flask, render_template
import pandas as pd

from assignment5 import Assignment5
from assignment9 import Assignment9
from assignment10 import Assignment10
from assignment11 import Assignment11
from assignment12 import Assignment12
from assignment16 import Assignment16
from assignment17 import Assignment17
import plotly.express as px
import plotly
import json

app = Flask(__name__)


# home page
@app.route("/")
def home():
    return render_template("index.html.j2")


# 404 handler
@app.errorhandler(404)
def not_found(e):
    return render_template("404.html.j2")


# first method - kept it simple here, it is defined right here within the file
@app.route("/assignment4")
def assignment4():
    filename = os.path.join(app.root_path, 'data', '4_tax2gdp.csv')
    tax2gdp = pd.read_csv(filename)

    # filter some outliers
    tax2gdp2 = tax2gdp[tax2gdp['GDP (In billions)'] < 10000]

    fig = px.bar(x=tax2gdp2["Tax Percentage"],
                 y=tax2gdp2["GDP (In billions)"]
                 )
    fig.update_layout(
        title='Tax rate by GDP for countries. Still WIP. Need to figure out how to add the country name on hover.',
        showlegend=True)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("assignment4.html.j2", graphJSON=graphJSON)


# second method - this is defined in its own file and we just call the method
@app.route("/assignment5")
def assignment5():
    obj = Assignment5()
    return obj.process()


# ninth assignment - static function used
@app.route("/assignment9")
def assignment9():
    return Assignment9.process()


@app.route("/assignment10")
def assignment10():
    return Assignment10.process()


@app.route("/assignment11")
def assignment11():
    return Assignment11.process()


@app.route("/assignment12")
def assignment12():
    return Assignment12.process()


@app.route("/assignment16")
def assignment16():
    return Assignment16.process()


@app.route("/assignment17")
def assignment17():
    return Assignment17.process()


# background process happening without any refreshing
@app.route('/assignment10_predict', methods=['POST'])
def assignment10_predict():
    return Assignment10.predict()


if __name__ == "__main__":
    app.run(debug=True)
