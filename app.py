import re
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import plotly.express as px
import pandas as pd
import numpy as np
import nltk
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from gensim.models import word2vec

df = pd.read_excel("C:/Users/rlope/Documents/Data for Projects/Hotel/processed hotel reviews.xlsb", engine='pyxlsb')
df['pos_tokens'] = df['pos_tokens'].apply(lambda x: [re.sub(' ','_',i) for i in re.findall(r"'([^']*)'", x)])
country_list = df['Reviewer_Nationality'].unique()
country_list.sort()

df_copy = df.copy()
df_explode = df_copy.explode('pos_tokens')


def td_idf(data, group, token):
    # getting word count in each group
    counts = data.groupby(group)[token].value_counts().to_frame().rename(columns={token: 'n_w'})

    # each group's total word count
    word_sum = counts.groupby(level=0).sum().rename(columns={'n_w': 'n_d'})

    # adding each group's total word count to word frequency
    tf = counts.join(word_sum)

    tf['tf'] = tf.n_w / tf.n_d

    c_d = counts.index.get_level_values(0).nunique()

    idf = counts.reset_index().groupby(token)[group].nunique().to_frame().rename(columns={group: 'i_d'}).sort_values(
        'i_d')

    idf['idf'] = np.log(c_d / idf.i_d.values)
    # idf = idf.reset_index()

    tf_idf = tf.join(idf)
    # tf_idf = tf.merge(idf, on=token, how='left')

    tf_idf['tf_idf'] = tf_idf.tf * tf_idf.idf

    tf_idf = tf_idf.reset_index()

    return tf_idf


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

TOP_FREQUENCY_COMPS = [
    dbc.CardHeader(html.H5("Comparison of most frequently used words for two nationalities")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-bigrams-comps",
                children=[
                    dbc.Alert(
                        "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                        id="no-data-alert-bigrams_comp",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(html.P("Choose two nationalities to compare:"), md=12),
                            dbc.Col(
                                [
                                    dcc.Dropdown(
                                        id="country_1",
                                        options=[
                                            {"label": i, "value": i}
                                            for i in country_list
                                        ],
                                        value="United Kingdom",
                                    )
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    dcc.Dropdown(
                                        id="country_2",
                                        options=[
                                            {"label": i, "value": i}
                                            for i in country_list
                                        ],
                                        value="United States of America",
                                    )
                                ],
                                md=6,
                            ),
                        ]
                    ),
                    dcc.Graph(id='top_10'),
                    dcc.Graph(id='top_idf'),
                    dcc.Graph(id='freq_scat'),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

BODY = dbc.Container(
    [
        dbc.Row([dbc.Col(dbc.Card(TOP_FREQUENCY_COMPS)), ], style={"marginTop": 30}),
    ],
    className="mt-12",
)

app.layout = html.Div(children=[BODY])


@app.callback(
    Output("top_10", "figure"),
    Output("top_idf", "figure"),
    Output("freq_scat", "figure"),
    Input("country_1", "value"),
    Input("country_2", "value"))
def country_comparisons(country_1, country_2):
    countries = [country_1, country_2]
    comparison = df_explode[df_explode['Reviewer_Nationality'].isin(countries)]

    national_comp = td_idf(comparison, 'Reviewer_Nationality', 'pos_tokens')

    top_tf = national_comp.sort_values(['Reviewer_Nationality', 'tf'], ascending=False).groupby(
        'Reviewer_Nationality').head(10)
    top_tfidf = national_comp.sort_values(['Reviewer_Nationality', 'tf_idf'], ascending=False).groupby(
        'Reviewer_Nationality').head(10)

    top_tf = top_tf.sort_values(['Reviewer_Nationality', 'tf'], ascending=True)
    top_tfidf = top_tfidf.sort_values(['Reviewer_Nationality', 'tf_idf'], ascending=True)

    countries = national_comp['Reviewer_Nationality'].unique()
    sub_set = national_comp[['Reviewer_Nationality', 'pos_tokens', 'tf']].copy()
    sub_set_reformat = sub_set.pivot_table(index='pos_tokens', columns='Reviewer_Nationality',
                                           values='tf').reset_index()

    top_10 = px.bar(top_tf, x="tf", y="pos_tokens", facet_col="Reviewer_Nationality",
                    title="Most Frequently Used Words", facet_col_spacing=0.1, hover_data=["tf"],
                    labels={'tf': '', 'pos_tokens': ''}).update_yaxes(matches=None, showticklabels=True, col=2)
    top_idf = px.bar(top_tfidf, x="tf_idf", y="pos_tokens", facet_col="Reviewer_Nationality",
                     title="Most Important Words", facet_col_spacing=0.1, hover_data=["tf_idf"],
                     labels={'tf_idf': '', 'pos_tokens': ''}).update_yaxes(matches=None, showticklabels=True, col=2)
    freq_scat = px.scatter(sub_set_reformat, x=countries[0], y=countries[1], trendline="ols", hover_name="pos_tokens")

    return (
        top_10,
        top_idf,
        freq_scat
    )

#http://localhost:8050/ if it doesn't connect
if __name__ == '__main__':
    app.run_server(debug=True)
