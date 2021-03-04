import pandas as pd
import numpy as np
import tensorflow as tf
import nltk

import logzero
import logging
from logzero import logger

def get_balanced_dataset(filepath, save_fp, y_column):
    file_type = filepath.split('.')[-1]
    logger.warn(f"Filetype is {file_type}")
    if file_type == 'gz':
        df = pd.read_csv(filepath, compression=compression, low_memory=False, 
                     nrows=nrows, parse_dates=['diner_date', 'rating_date'])
        df = df.head(20000)
        df = clean_reviews(df)
        return df
    elif file_type == 'json':
        df = pd.read_json(filepath, lines=True)
    
    df['usable_rating'] = df['rating'].apply(lambda r: int(r)-1)
    
    min_label = df[y_column].value_counts().min()
    balanced_df = (df.groupby(y_column)).sample(n=min_label, random_state=0)
    balanced_df.set_index(['review_id'], inplace=True)
    balanced_df.rename(columns={"comment": "review"}, inplace=True)
    balanced_df = split_reviews_per_sentence(balanced_df)

    balanced_df.to_csv(save_fp, sep='#', index_label='review_id')
    return balanced_df

# clean_content: ['we', 'are', 'having']
# review: 'we are having'
# review_sentences: ['we are.', 'having.']

def clean_reviews(reviews, colname='review'):
    reviews[colname] = reviews[colname].apply(lambda x: ' '.join(eval(x)))
    return reviews


def split_reviews_per_sentence(reviews, colname='review'):
    reviews['review_sentences'] = reviews[colname].apply(lambda rvw: nltk.sent_tokenize(rvw))
    return reviews


def preprocess(filepath):
    file_type = filepath.split('.')[-1]
    logger.warn(f"Filetype is {file_type}")
    if file_type == 'gz':
        df = pd.read_csv(filepath, compression=compression, low_memory=False, 
                     nrows=nrows, parse_dates=['diner_date', 'rating_date'])
        df = clean_reviews(df)
    elif file_type == 'json':
        df = pd.read_json(filepath, lines=True)
        df.rename(columns={"comment": "review"}, inplace=True)
    else:
        raise NotImplementedError(f"Only gz and json files are supported, found {file_type}")
    
    return split_reviews_per_sentence(df)