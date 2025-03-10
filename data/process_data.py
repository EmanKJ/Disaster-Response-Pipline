# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Description: This function load messages and categories data and marge them into one data frame called df

    Arguments:
    messages_filepath: the path of masseges file enter by the user. 
    categories_filepath: the path of categoriese file enter by the user. 

    Returns: data frame that marge both messages and categories

    """
    messages = pd.read_csv(messages_filepath)
    messages.head()
    #-------------------------------------
    categories = pd.read_csv(categories_filepath)
    categories.head()
    # merge datasets
    df = pd.merge(messages, categories, left_on='id', right_on='id', how='outer')
    df.head()
    return df

def clean_data(df):
    """
    Description: This function cleaning the data frame (df) by spliting categories, and removing the duplucation

    Arguments:
    df: data frame that consists of messages and categories data

    Returns: 
    df after clean it

    """
    categories = df.categories.str.split(';', expand = True)
    categories.head()
    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda i: i[:-2]).values.tolist()
    #print(category_colnames)
    # rename the columns of `categories`
    categories.columns = category_colnames
    #categories.head()
    
    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    #categories.head()
    #Replace categories column in df with new category columns
    # drop the original categories column from `df`
    df.drop('categories',axis=1, inplace = True)
    #df.head()
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    #Remove duplicates
    # drop duplicates
    df.drop_duplicates(subset = 'id', inplace = True)
    # check number of duplicates
    df.duplicated().sum()
    
    return df

def save_data(df, database_filename):
    """
    Description: This function save the cleaned dataframe (df) as a database
    

    Arguments:
    df: cleaned data frame that consists of messages and categories data
    database_filename : the name of the database specified by the user

    Returns: None

    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('CleanTable', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
    
