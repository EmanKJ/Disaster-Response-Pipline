import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from  sklearn.pipeline import make_pipeline



def load_data(databasePath):
        """
    Description: This function load the data form the database Specified  by the uset and split the data inot masseges (x), labels(y), and category_names

    Arguments:
    databasePath: the path of the database. 
 

    Returns: 
    x: masseges 
    Y: labels 
    category_names: names of the categories
    """
    engine = create_engine('sqlite:///'+databasePath)
    df = pd.read_sql_table('CleanTable', engine)
    X =  df.message.values
    Y = df[df.columns[4:]].values
    category_names = list(df.columns[4:])
    return X,Y,category_names

def tokenize(text):
    """
    Description: This function split the text massge into tokenizatiob inorder to analyze it

    Arguments:
    text: Message data for tokenization.
 

    Returns: 
    clean_tokens: Result list after tokenization
    """
    clean_tokens = []
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    


def build_model():
    """
    Description: This function build the model which will use latter ot predict the message, berforme gride search to find the best parameters

    Arguments:
    None
    Returns: 
    cv: the result of GridSearch of the model
    """
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])
    parameters = {'clf__estimator__n_estimators': [50,20],
                  'clf__estimator__min_samples_split': [50,20],
                  'clf__estimator__criterion': ['entropy', 'gini']
                 }
        
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Description: This function evaluate the model returened by previous function and print the evaluation matric results

    Arguments:
    model
    X_test: the masseges in test dataset 
    Y_test: the labels in test dataset 
    category_names
    Returns: 
    None
    """
    prediction = model.predict(X_test)
    print(classification_report(Y_test, prediction, target_names = category_names))
    print('---------------------------------')
    for i in range(Y_test.shape[1]):
        print('%25s accuracy : %.2f' %(category_names[i], accuracy_score(Y_test[:,i], prediction[:,i])))
        print('%25s accuracy : %.2f' %(category_names[i],recall_score(Y_test[:,i], prediction[:,i], average='weighted')))
        print('%25s accuracy : %.2f' %(category_names[i],precision_score(Y_test[:,i], prediction[:,i], average='weighted')))
        print('%25s accuracy : %.2f' %(category_names[i],f1_score(Y_test[:,i], prediction[:,i], average='weighted')))
   


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        print(model)
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