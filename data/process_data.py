import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
        - messages_filepath: filepath to the messages csv data
        
        - categories_filepath: filepath to the categories csv data
    
    OUTPUT:
        - df: DataFrame of the two files merged
    '''
    
    # load
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge
    df = messages.merge(categories, on='id', how='left')
    
    return df

def clean_data(df):
    '''
    INPUT:
        - df: DataFrame containing merged data from load_data
        
    OUTPUT:
        - df: DataFrame of cleaned data
    '''
    
    # split categories into separate columns along the ';' marker (use expand=True)
    categories = df['categories'].str.split(pat=';', expand=True)

    # use first row to create column names
    # separate row
    row = categories.iloc[0]

    # apply lambda function with slicing to remove the hyphenated numbers
    category_colnames = row.apply(lambda x: x[:-2])

    # rename columns in categories
    categories.columns = category_colnames

    # convert categories to number answer only
    for column in categories:
        # set value to be last character (number answer)
        categories[column] = categories[column].str[-1]
        
        # convert from string to numeric
        categories[column] = categories[column].astype(int)
        
    
    '''
    Issue: additional cleaning: some values in the categories aren't a 0 or 1
    
    Solution: any data not a 0 or 1 becomes a 1
    
    category_error(col) returns a column where any value not 0 or 1 is assumed
    to be a 1.
    '''
    def category_error(col):
        for count, element in enumerate(col):
            if element not in [0, 1]:
                col[count] = 1
        return col

    categories = categories.apply(lambda col: category_error(col))    
    # drop categories column in df and replace with the categories df
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)

    # remove duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    '''
    INPUT:
        - df: DataFrame of cleaned data
        
        - database_filename: filepath of where to save the database
        
    OUTPUT:
        - saves a SQLite database in desired location
    '''
    
    # save to a sqlite database using sqlalchemy
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False)  


def main():
    '''
    Function strings together an entire ETL pipeline
    '''
    
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