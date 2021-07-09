from flask import render_template
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly
import json

from customutils import CustomUtils
import warnings; warnings.simplefilter('ignore')


class Assignment17:
    @staticmethod
    def process():
        df = CustomUtils.read_file_and_return_df('17_monthly_ridership.csv')
        # df.head()

        # rename the column names
        df.columns = ["month", "average_monthly_ridership"]
        # df.head()

        # data cleanup
        df['average_monthly_ridership'].unique()
        df = df.drop(df.index[df['average_monthly_ridership'] == ' n=114'])

        # correct the column dtypes
        df['average_monthly_ridership'] = df['average_monthly_ridership'].astype(np.int32)
        df['month'] = pd.to_datetime(df['month'], format='%Y-%m')
        # df.dtypes

        average_rider_line_chart = px.line(df, x="month", y="average_monthly_ridership", title='Average monthly bus riders in Oergon', height=600)

        # change the month to numeric format so we have monthly data rather than yearly
        to_plot_monthly_variation = df
        mon = df['month']
        temp = pd.DatetimeIndex(mon)
        month = pd.Series(temp.month)
        to_plot_monthly_variation = to_plot_monthly_variation.drop(['month'], axis=1)
        to_plot_monthly_variation = to_plot_monthly_variation.join(month)
        to_plot_monthly_variation.head()

        average_rider_bar_chart = px.bar(to_plot_monthly_variation, x='month', y='average_monthly_ridership', height=600)

        # observations = ridership declines in july and august
        # Applying Seasonal ARIMA model to forcast the data
        mod = sm.tsa.SARIMAX(df['average_monthly_ridership'], trend='n', order=(0, 1, 0), seasonal_order=(1, 1, 1, 12))
        results = mod.fit()
        # print(results.summary())

        df['forecast'] = results.predict(start=102, end=120, dynamic=True)
        rider_forecast = px.line(df, x='month', y=['average_monthly_ridership', 'forecast'], height=600)

        return render_template("assignment17.html.j2",
                               sample_dataset=df.head(5).to_html(classes='table table-striped', index=False, justify='center'),
                               average_rider_line_json=json.dumps(average_rider_line_chart, cls=plotly.utils.PlotlyJSONEncoder),
                               average_rider_bar_json=json.dumps(average_rider_bar_chart, cls=plotly.utils.PlotlyJSONEncoder),
                               rider_forecast_json=json.dumps(rider_forecast, cls=plotly.utils.PlotlyJSONEncoder)
                               )

    @staticmethod
    def get_recommendations(indices, cosine_sim, titles, title):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:31]
        movie_indices = [i[0] for i in sim_scores]
        return titles.iloc[movie_indices].to_frame()
