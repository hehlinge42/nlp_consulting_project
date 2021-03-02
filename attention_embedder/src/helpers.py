import pandas as pd
import numpy as np
import tensorflow as tf
import nltk

def stratify_data(original_df, y_column):
    min_label = original_df[y_column].value_counts().min()
    df = pd.DataFrame(columns=original_df.columns)
    for label in original_df[y_column].unique():
      subdf = original_df[original_df[y_column] == label][:min_label]
      df = df.append(subdf)
    return df


def get_reviews(filepath, compression='gzip', nrows=None):
    return pd.read_csv(filepath, compression=compression, low_memory=False, 
                     nrows=nrows, parse_dates=['diner_date', 'rating_date'])


def clean_reviews(reviews, colname='review'):
    reviews[colname] = reviews.content.apply(lambda x: ' '.join(eval(x)))
    return reviews


def split_reviews_per_sentence(reviews, colname='review_sentences'):
    reviews[colname] = reviews.review.progress_apply(
        lambda rvw: nltk.sent_tokenize(rvw))
    return reviews