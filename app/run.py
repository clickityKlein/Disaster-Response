import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    # plot 1: bar chart of messages by genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # plot 2: bar chart of top 10 categoreis
    categories = df.iloc[:,4:]
    category_hits = pd.DataFrame([[category, sum(categories[category])] for category in categories])
    top_hits = category_hits.sort_values(1, ascending=False).iloc[:10]
    
    # plot 3: bar chart of number of category assignments per message
    message_hits = categories.sum(axis=1).value_counts()
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # plot 1
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Messages"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # plot 2
        {
            'data': [
                Bar(
                    x=top_hits.iloc[:,0],
                    y=top_hits.iloc[:,1]
                )
            ],

            'layout': {
                'title': 'Top 10 Categories',
                'yaxis': {
                    'title': "Messages"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        # plot 3
        {
            'data': [
                Bar(
                    x=message_hits.index,
                    y=message_hits
                )
            ],

            'layout': {
                'title': 'Categories per Message',
                'yaxis': {
                    'title': "Messages"
                },
                'xaxis': {
                    'title': "Number of Categories"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()