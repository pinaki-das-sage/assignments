from flask import render_template
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from customutils import CustomUtils
import warnings; warnings.simplefilter('ignore')


class Assignment16:
    @staticmethod
    def process():
        md = CustomUtils.read_file_and_return_df('16_movies_metadata.csv')
        # md.head()

        # fill the null values with []
        md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(
            lambda x: [i['name'] for i in x] if isinstance(x, list) else []
        )

        # get the vote counts and averages for all movies
        vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
        vote_mean = vote_averages.mean()
        # vote_mean

        top_vote_counts = vote_counts.quantile(0.95)
        # top_vote_counts

        # get release year for all movies in a new column
        md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(
            lambda x: str(x).split('-')[0] if x != np.nan else np.nan
        )

        # get the above average movies list
        qualified = md[(md['vote_count'] >= top_vote_counts) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][
            ['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')
        # qualified.shape

        # get the top 250 movies by vote average
        qualified = qualified.sort_values('vote_average', ascending=False).head(250)
        # qualified.head(15)

        s = md.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'genre'
        gen_md = md.drop('genres', axis=1).join(s)

        best_romantic_movies = Assignment16.build_chart(gen_md, 'Romance').head(15)

        links_small = CustomUtils.read_file_and_return_df('16_links_small.csv')
        links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

        md = md.drop([19730, 29503, 35587])
        md['id'] = md['id'].astype('int')
        smd = md[md['id'].isin(links_small)]
        # smd.shape

        smd['tagline'] = smd['tagline'].fillna('')
        smd['description'] = smd['overview'] + smd['tagline']
        smd['description'] = smd['description'].fillna('')

        tf = TfidfVectorizer(analyzer='word')
        tfidf_matrix = tf.fit_transform(smd['description'])
        # tfidf_matrix.shape

        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        # cosine_sim[0]

        smd = smd.reset_index()
        titles = smd['title']
        indices = pd.Series(smd.index, index=smd['title'])

        movie_to_search = 'Batman Begins'
        recommendations = Assignment16.get_recommendations(indices, cosine_sim, titles, movie_to_search).head(10)

        return render_template("assignment16.html.j2", vote_counts=vote_counts, vote_averages=vote_averages,
                                vote_mean=vote_mean, best_romantic_movies=best_romantic_movies.to_html(classes='table table-striped', index=False, justify='center'),
                                movie_to_search=movie_to_search, recommendations=recommendations.to_html(classes='table table-striped', index=False, justify='center'),
                                sample_dataset=md.head(5).to_html(classes='table table-striped', index=False, justify='center')
                               )

    @staticmethod
    def build_chart(gen_md, genre, percentile=0.85):
        df = gen_md[gen_md['genre'] == genre]
        vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
        C = vote_averages.mean()
        m = vote_counts.quantile(percentile)

        qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][
            ['title', 'year', 'vote_count', 'vote_average', 'popularity']]
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')

        qualified['wr'] = qualified.apply(
            lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) + (m / (m + x['vote_count']) * C),
            axis=1)
        qualified = qualified.sort_values('wr', ascending=False).head(250)

        return qualified

    @staticmethod
    def get_recommendations(indices, cosine_sim, titles, title):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:31]
        movie_indices = [i[0] for i in sim_scores]
        return titles.iloc[movie_indices].to_frame()
