import re
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

df = pd.read_excel("/home/rlopezra/mysite/data/cleaned reviews.xlsb", engine='pyxlsb')
df['pos_tokens'] = df['pos_tokens'].apply(lambda x: [re.sub(' ','_',i) for i in re.findall(r"'([^']*)'", x)])

country_list = df['Reviewer_Nationality'].unique()
country_list.sort()
df_copy = df.copy()
df_explode = df_copy.explode('pos_tokens')

#Word2Vect corpus and model
model = KeyedVectors.load_word2vec_format("/home/rlopezra/mysite/data/model.bin", binary=True, unicode_errors='ignore')

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

def tsne_data(model, compl):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=compl, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    df = pd.DataFrame({"token": labels,
                       "x": x,
                       "y": y})

    return df


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

NAVBAR = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(
                        dbc.NavbarBrand("Ronald Lopez's Plotly NLP Dashboard", className="ml-2")
                    ),
                ],
                align="center",
                no_gutters=True,
            ),
            href="https://plot.ly",
        )
    ],
    color="dark",
    dark=True,
    sticky="top",
)


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
                                        value=" Belgium ",
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
                                        value=" United States of America ",
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

Word2Vec = [
    dbc.CardHeader(html.H5("Displaying a Word Embedding in Two-Dimensional Space")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-bigrams-scatter",
                children=[
                    dbc.Alert(
                        "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                        id="no-data-alert-bigrams",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(html.P(["Choose a t-SNE perplexity value:"]), md=5),
                            dbc.Col(
                                [
                                    dcc.Slider(
                                        id='complex-slider',
                                        min=50,
                                        max=700,
                                        value=50,
                                        marks={str(i): str(i)  for i in range(50, 750, 50)},
                                        step=None
                                    )
                                ],
                                md=10,
                            ),
                        ]
                    ),
                    dcc.Graph(id="w2v-scatter"),

                    html.Div([
                        dcc.Markdown("""
                            **Click on a point to see the ten closest use words **

                        """),
                        html.Ul(id='my-list'),
                    ]),
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
        dbc.Row([dbc.Col(dbc.Card(Word2Vec)), ], style={"marginTop": 30})
    ],
    className="mt-12",
)


app.layout = html.Div(children=[NAVBAR, BODY])

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
                    title="Most Frequently Used Words", facet_col_spacing=0.15, hover_data=["tf"],
                    labels={'tf': '', 'pos_tokens': ''}).update_yaxes(matches=None, showticklabels=True, col=2)
    top_idf = px.bar(top_tfidf, x="tf_idf", y="pos_tokens", facet_col="Reviewer_Nationality",
                     title="Most Important Words", facet_col_spacing=0.15, hover_data=["tf_idf"],
                     labels={'tf_idf': '', 'pos_tokens': ''}).update_yaxes(matches=None, showticklabels=True, col=2)
    freq_scat = px.scatter(sub_set_reformat, x=countries[1], y=countries[0], trendline="ols", hover_name="pos_tokens")

    return (
        top_10,
        top_idf,
        freq_scat
    )

@app.callback(
    Output("w2v-scatter", "figure"),
    Input("complex-slider", "value"))
def w2v_update(complexity):
    w2v_df = tsne_data(model, complexity)

    w2v_scat = px.scatter(w2v_df, x='x', y='y', hover_name="token", labels={'x':' ', 'y':' '})

    return w2v_scat

@app.callback(
    Output('my-list', 'children'),
    Input('w2v-scatter', 'clickData'))
def display_click_data(clickData):
    for i in clickData['points']:
        word = i['hovertext']

    result = model.wv.similar_by_word(word)
    similar_list = [word[0] for word in result]

    return html.Ul([html.Li(x) for x in similar_list])

if __name__ == '__main__':
    app.run_server(debug=True)
