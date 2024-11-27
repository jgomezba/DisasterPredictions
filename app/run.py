import json
import plotly
import pandas as pd

from googletrans import Translator

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def translate_to_english(text):
    # Initialize the translator
    translator = Translator()

    # Translate the text to English
    translated = translator.translate(text, dest='en')

    if translated is None:
        translated = text
    
    return translated.text

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('data', engine)

# load model
model = joblib.load("models/random_forest_tuned.pkl")
model_binary = joblib.load("models/multi_label.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    department_counts = df.groupby('departments_affected').count()['message']
    department_names = list(department_counts.index)
    
    department_names_ordered = ['Three or less', 'Five or less', 'More than five']
    department_names_sorted = sorted(department_names, key=lambda x: department_names_ordered.index(x))
    department_counts_sorted = department_counts[department_names_sorted]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
    {
        'data': [
            Scatter(
                x=department_names_sorted,
                y=department_counts_sorted,
                mode='markers+lines',  # Show both lines and markers for a modern effect
                marker=dict(
                    size=12,  # Size of the markers
                    color=department_counts,  # Color by count, for a color gradient
                    colorscale='Viridis',  # Modern color scale
                    showscale=True,  # Show color scale
                    line=dict(color='#2ecc71', width=2),  # Add a border to the markers for emphasis
                    opacity=0.8  # Slightly transparent markers for a modern look
                ),
                line=dict(
                    width=2,  # Line width for the connecting lines
                    color='#2ecc71'  # Line color matching the marker border
                ),
            )
        ],

        'layout': {
            'title': {
                'text': 'Distribution of Departments Affected',
                'x': 0.5,  # Center the title
                'font': {
                    'size': 24,
                    'family': 'Arial, sans-serif',
                    'color': '#2c3e50'
                }
            },
            'xaxis': {
                'tickangle': -45,  # Rotate x-axis labels for better visibility
                'showgrid': True,
                'gridcolor': '#ecf0f1',
                'showline': True,
                'linewidth': 2,
                'linecolor': '#bdc3c7'
            },
            'yaxis': {
                'title': {
                    'text': 'Count',
                    'font': {
                        'size': 16,
                        'family': 'Arial, sans-serif',
                        'color': '#34495e'
                    }
                },
                'showgrid': True,
                'gridcolor': '#ecf0f1',
                'showline': True,
                'linewidth': 2,
                'linecolor': '#bdc3c7'
            },
            'plot_bgcolor': '#f4f6f7',
            'paper_bgcolor': '#ffffff',
            'bargap': 0.15,  # Introduce space between markers
            'hovermode': 'closest',
            'autosize': True,
            'margin': {
                'l': 60, 'r': 60, 'b': 80, 't': 100
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
    classification_labels = model.predict([translate_to_english(query)])
    new_labels = model_binary.inverse_transform(classification_labels)
    
    categories = model_binary.classes_
    classification_labels = new_labels[0]
    
    classification_results = {category: 1 if category in classification_labels else 0 for category in categories}

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()