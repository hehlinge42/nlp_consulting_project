import pandas as pd
import numpy as np
import tensorflow as tf
import nltk

import logzero
import logging
from logzero import logger

def gen_balanced_df(filepath, save_fp, y_column, balance):
    
    file_type = filepath.split('.')[-1]
    logger.warn(f"Filetype is {file_type}")
    if file_type == 'gz':
        df = pd.read_csv(filepath, compression='gzip', low_memory=False, 
                     nrows=20000, parse_dates=['diner_date', 'rating_date'])
        df.rename(columns={"content": "review"}, inplace=True)
        df = clean_reviews(df)
    elif file_type == 'json':
        df = pd.read_json(filepath, lines=True)
        df.rename(columns={"comment": "review"}, inplace=True)

    df['usable_rating'] = df['rating'].apply(lambda r: int(r)-1)
    df = split_reviews_per_sentence(df)
    # Balanced dataset
    if balance is True:
        min_label = df[y_column].value_counts().min()
        df = (df.groupby(y_column)).sample(n=min_label, random_state=0)
        df.set_index(['review_id'], inplace=True)
        df.to_csv(save_fp, sep='#', index_label='review_id')

    return df


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