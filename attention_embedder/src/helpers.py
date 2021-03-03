import pandas as pd
import numpy as np
import tensorflow as tf
import nltk

import logzero
import logging
from logzero import logger

def stratify_data(original_df, y_column):
    min_label = original_df[y_column].value_counts().min()
    df = pd.DataFrame(columns=original_df.columns)
    for label in original_df[y_column].unique():
      subdf = original_df[original_df[y_column] == label][:min_label]
      df = df.append(subdf)
    return df

    # reviews = pd.read_json(os.path.join(scraped_data_dir, 'merged_data', 'merged_reviews.json'), lines=True)
    # by_rating = reviews.groupby(by=['rating']).count()
    # min_count = min(by_rating['review_id'])
    # balanced_reviews = (reviews.groupby("rating")).sample(n=min_count, random_state=0)
    # balanced_reviews.set_index(['review_id'], inplace=True)
    # balanced_reviews.to_csv(os.path.join(scraped_data_dir, 'merged_data', 'balanced_reviews.csv'), sep='#', index_label='review_id')

# clean_content: ['we', 'are', 'having']
# review: 'we are having'
# review_sentences: ['we are.', 'having.']

def clean_reviews(reviews, colname='review'):
    reviews[colname] = reviews[colname].apply(lambda x: ' '.join(eval(x)))
    return reviews


def split_reviews_per_sentence(reviews, colname='review'):
    reviews['review_sentences'] = reviews[colname].progress_apply(
        lambda rvw: nltk.sent_tokenize(rvw))
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