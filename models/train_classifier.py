import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

def load_data(database_filepath):
    '''
    INPUT:
        - database_filepath: filepath to where SQLite database is saved
        
    OUTPUT:
        - X: independent Series of data (messages)
        
        - y: dependent DataFrame of data (categories)
        
        - y.columns: Series containing the names of the categories
    '''
    
    # load data from database
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    y = df.iloc[:,4:]
    
    return X, y, y.columns


def tokenize(text):
    '''
    INPUT:
        - text: an individual message
        
    OUTPUT:
        - clean_tokens: message after the tokenization process, returned as a list
    '''
    
    # remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    OUTPUT:
        - cv: GridSearchCV object containing a pipeline, parameters for the grid search,
        and other GridSearchCV object specific parameters
    '''
    
    # pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    # parameters
    parameters = {
        'clf__estimator__criterion': ['gini', 'entropy', 'log_Loss'],
        'clf__estimator__n_estimators': [50, 100, 200]
        }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=5, verbose=2, cv=2)
    return cv


'''
Basic build_model() function included below for testing purposes (shorter run-time)
'''
# def build_model():
#     # pipeline
#     pipeline = Pipeline([
#         ('vect', CountVectorizer(tokenizer=tokenize)),
#         ('tfidf', TfidfTransformer()),
#         ('clf', RandomForestClassifier())
#         ])
    
#     return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
        - model: object resulting from the training process
        
        - X_test: testing split of the message data
        
        - Y_test: testing split of the categories data
        
        - category_names: column names of the categories data
        
    OUTPUT:
        - classification_report: printed report of model performance across the categories,
        scored by precision, recall and f1-score.
    '''
    
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names, zero_division=1))


def save_model(model, model_filepath):
    '''
    INPUT:
        - model: object resulting from the training process
        
        - model_filepath: filepath of where to save the model file
        
    OUTPUT:
        - saves a pickle file of the model in desired location
    '''
    
    with open ('model.pkl', 'wb') as f:
        pickle.dump(model, f)


def main():
    '''
    Function strings together an entire ETL pipeline
    '''
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()