import sys

from sqlalchemy import create_engine
import nltk
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import pickle

nltk.download(['punkt', 'wordnet', 'stopwords','omw-1.4'])


def load_data(database_filepath):
    """
    Arg: database path to SQL
    Out: X values, y values and category names
    
    """
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('Disasters', engine)
    X = df['message']
    y = df[df.columns[4:]]
    category_names = y.columns
    return X, y, category_names 

def tokenize(text):
    
    """
    Arg: text that to be tokenized
    Out: list of tokens 
    
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens    


def build_model():
    """
    Return a Machine Learn model that process the text messages
    
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        "clf__estimator__min_samples_split": [2, 6],
        "clf__estimator__max_depth": [4, 6, 8]}
    
    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model, X_test, y_test, category_names):  
    
    """
    Args: model > machine learn model
    X_test > a sample of X values to be used to predict y values as y_pred
    y_test > a sample of real y values 
    category_names > category labels
    
    """
    y_pred = model.predict(X_test)   
    
    for i, col in enumerate(y_test):
        print(col)
        print(i)
        print(classification_report(y_test[col], y_pred[:, i]))

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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