import sys
import pandas as pd
import sqlite3


def clean_categories(text:str):
    result = []
    text_splitted = text.split(";")
    for category in text_splitted:
        category_values = category.split("-")
        if category_values[1] == "1" and category_values[0]!="related":
            result.append(category_values[0])
            
    return result

def count_departments_affected(list_items:list):
    departments_affected = len(list_items)
    if departments_affected <= 3:
        return "Three or less"
    elif departments_affected <= 5:
        return "Five or less"
    else:
        return "More than five"

def load_data(messages_filepath, categories_filepath):
    df_categories = pd.read_csv(categories_filepath)
    df_messages = pd.read_csv(messages_filepath)
    
    df_result = df_messages.merge(df_categories, on=["id"], how = "inner")
    
    return df_result

def clean_data(df):
    df["categories_proccessed"] = df["categories"].apply(clean_categories)
    df["departments_affected"] = df["categories_proccessed"].apply(count_departments_affected)
    for col in df.columns:
        if df[col].apply(type).isin([list, dict, set]).any():
            df[col] = df[col].apply(str)
    
    df = df[df["categories_proccessed"]!="[]"]
    return df

def save_data(df, database_filename):
    conn = sqlite3.connect(database_filename)
    try:
        df.to_sql("data", conn, if_exists='replace', index=False)
    except Exception as e:
        print(f"Error saving dataframe in database: {e}")
    finally:
        conn.close()  


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