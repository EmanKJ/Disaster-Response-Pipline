import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.externals import joblib

def load_data(databasePath):
    engine = create_engine(databasePath)
    df = pd.read_sql_table('CleanTable', engine)
    X =  df.message.values
    Y = df[df.columns[4:]].values
    category_names = list(df.columns[4:])
    return X,Y,category_names

def tokenize(text):
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokenst
    


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
                'bootstrap': [True, False],
                'n_estimators':[20,50,100,150],
                'max_depth': [2, 5,20,50]
             }
        
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='precision_samples', cv = 6)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
   prediction = model.predict(X_test)
    print(classification_report(Y_test, prediction, target_names = category_names))
    print('---------------------------------')
    for i in range(Y_test.shape[1]):
        print('%25s accuracy : %.2f' %(category_names[i], accuracy_score(Y_test[:,i], prediction[:,i])))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()

        
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