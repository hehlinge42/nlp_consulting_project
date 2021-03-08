import pandas as pd
import numpy as np
import tensorflow as tf
import nltk

import os

import logzero
import logging
from logzero import logger


def character_transformer(document):
    with_accent = ['é', 'è', 'à', "ê", "\u2019"]
    without_accent = ['e', 'e', 'a', "e", "'"]
    transformation_dict = {before:after for before, after in zip(with_accent, without_accent)}
    return document.translate(str.maketrans(transformation_dict))


def unicode_remover(document):
    encoded = document.encode('ascii', 'replace')
    return encoded.decode('utf-8')


def character_remover(document):
    characters_to_remove = ["@", "/", "#", ".", ",", "!", "?", 
                            "(", ")", "-", "_","’","'", "\"", 
                            ":", "\n", "\t", "\r"]
    transformation_dict = {initial: " " for initial in characters_to_remove}
    return document.translate(str.maketrans(transformation_dict))


def contraction_transformer(document, filename):
    with open(filename) as contractions:
        for word in document.split():
            if word in contractions:
                document = document.replace(word, contractions[word])
    return document


def clean_df(df, column, contraction_fp):

    df[column] = df[column].map(lambda x: x.lower())
    df[column] = df[column].map(lambda x: contraction_transformer(x, contraction_fp))
    df[column] = df[column].map(lambda x: character_transformer(x))
    df[column] = df[column].map(lambda x: unicode_remover(x))
    df[column] = df[column].map(lambda x: character_remover(x))

    return df

def gen_balanced_df(filepath, save_fp, y_column, balance):
    
    file_type = filepath.split('.')[-1]
    contraction_fp = os.path.join('cleaner', 'assets', 'contractions.json')

    logger.warn(f"Filetype is {file_type}")
    if file_type == 'gz':
        df = pd.read_csv(filepath, compression='gzip', low_memory=False, parse_dates=['diner_date', 'rating_date'])
        df.rename(columns={"content": "review"}, inplace=True)
        df = clean_reviews(df)
    elif file_type == 'json':
        df = pd.read_json(filepath, lines=True)
        df.rename(columns={"comment": "review"}, inplace=True)
        df = clean_df(df, 'review', contraction_fp)
    elif file_type == 'csv':
        df = pd.read_csv(filepath, low_memory=False, parse_dates=['diner_date', 'rating_date'])
        df.rename(columns={"content": "review"}, inplace=True)
        df = clean_reviews(df)

    df['usable_rating'] = df['rating'].apply(lambda r: int(r)-1)
    df = split_reviews_per_sentence(df)

    # Balanced dataset
    if balance is True:
        min_label = df[y_column].value_counts().min()
        min_label = min(min_label, 8000)
        df = (df.groupby(y_column)).sample(n=min_label, random_state=0)
        try:
            df.set_index(['review_id'], inplace=True)
        except:
            logger.error("No review id")
        df.to_csv(save_fp, sep='#', index=True)

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