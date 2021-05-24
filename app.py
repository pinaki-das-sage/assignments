import os

from flask import Flask, render_template
import pandas as pd

from assignment5 import Assignment5
from assignment9 import Assignment9

import plotly.express as px
import plotly
import json

app = Flask(__name__)


# home page
@app.route("/")
def home():
    return render_template("index.html.j2")


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



if __name__ == "__main__":
    app.run(debug=True)
