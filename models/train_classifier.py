import sys
import os

import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import spacy


# Load tokenizer with spacy
nlp = spacy.load('en_core_web_sm')

def load_data(database_filepath, table_name):
    try:
        conn = sqlite3.connect(database_filepath)
        cursor = conn.cursor()
        query = "SELECT message, categories_proccessed FROM " + table_name
        cursor.execute(query)
        rows = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        df = pd.DataFrame(rows, columns=column_names)
        df[column_names[1]] = ensure_list_of_lists(df, column_names[1])
    except sqlite3.Error as e:
        print(f"Error while reading data {e}")
    finally:
        if conn:
            conn.close()
    
    return df

def tokenize(texts):
    docs = nlp.pipe(texts, disable=["ner", "parser"])
    tokens_list = []
    for doc in docs:
        tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
        tokens_list.append(" ".join(tokens))
    return tokens_list

def get_pipeline():
    pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),  # Convierte el texto en características numéricas
            ('clf', OneVsRestClassifier(RandomForestClassifier(random_state=42, n_jobs=-1)))  # Clasificación multietiqueta
        ])
    return pipeline

def train_model_with_gridsearch(pipeline, X_train, y_train):
    # Parameters to adjust
    param_grid = {
        'tfidf__max_features': [1000, 2000],  # Número máximo de características
        'tfidf__ngram_range': [(1, 1), (1, 2)],  # Unigramas o bigramas
        'clf__estimator__n_estimators': [100, 200],  # Número de árboles en RandomForest
        'clf__estimator__max_depth': [10, 20, None],  # Profundidad máxima de los árboles
        'clf__estimator__min_samples_split': [2, 5]  # Número mínimo de muestras para dividir un nodo
    }

    # Initiliaze GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print(f"Mejores parámetros: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, categories):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=categories))

def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)

def multi_label_binarizer(dataframe: pd.DataFrame):
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(dataframe['categories_proccessed'])
    return mlb, y

def predict_new_cases(model, binary_model, new_messages):
    new_y_pred = model.predict(new_messages)
    new_labels = binary_model.inverse_transform(new_y_pred)
    i=0
    for mess in new_messages:
        print()
        print(f"Message: {mess}")
        print(f"Categories predicted {new_labels[i]}")
        i += 1

    return new_labels

def ensure_list_of_lists(df, column_name):
    df[column_name] = df[column_name].apply(lambda x: eval(x) if isinstance(x, str) else x)
    return df[column_name]

def main():
    if len(sys.argv) == 4:
        database_filepath, model_filepath, multi_label_filepath = sys.argv[1:]
        random_forest_model = os.path.join(os.path.dirname(__file__), model_filepath)
        multi_label_model = os.path.join(os.path.dirname(__file__), multi_label_filepath)
        
        if os.path.exists(random_forest_model) and os.path.exists(multi_label_model):
            print(f'Loading previous models generated:\n\t{random_forest_model}\n\t{multi_label_model}')
            random_forest_model_read = joblib.load(random_forest_model)
            multi_label_model_read = joblib.load(multi_label_model)
            print('Testing results...')
            predict_new_cases(random_forest_model_read, multi_label_model_read, new_messages = ["Heavy rain in the city", 
                                                        "We need water and medical supplies", 
                                                        "Please is cold here and I need some shoes",
                                                        "There is fire in my house and in road"])
        else:
            print('Loading data...\n    DATABASE: {}'.format(database_filepath))
            df = load_data(database_filepath, "data")

            print('Tokenizing data...')
            df['message_tokenized'] = tokenize(df['message'])
            
            print('Binarizing categories...')
            mlb, y = multi_label_binarizer(df)
            
            print('Dividing data into train and test...')
            X_train, X_test, y_train, y_test = train_test_split(df['message_tokenized'], y, test_size=0.2, random_state=42)

            print('Training model with GridSearchCV...')
            model = train_model_with_gridsearch(get_pipeline(), X_train, y_train)

            print('Evaluating model...')
            evaluate_model(model, X_test, y_test, mlb.classes_)

            print('Saving model...')
            save_model(model, model_filepath)
            save_model(mlb, multi_label_filepath)
            
            print('Trying new cases')
            predict_new_cases(model, mlb, new_messages = ["Heavy rain in the city", 
                                                        "We need water and medical supplies", 
                                                        "Please is cold here and I need some shoes",
                                                        "There is fire in my house and in road"])
            
    else:
        print('Please provide the path to the disaster messages database '
            'as the first argument and the path to the pickle file to '
            'save the model as the second argument and the multilabel model as third argument. \n\nExample: python '
            'train_classifier.py ../data/DisasterResponse.db random_forest_tuned.pkl multi_label.pkl')


if __name__ == '__main__':
    main()