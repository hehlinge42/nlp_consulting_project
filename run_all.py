
from scraper.merger import merge_files
from cleaner.src.cleaner import Cleaner
from embedder.src.embedder import Embedder
from cleaner.src.helpers import save_wordcloud, save_tfidf

import os
import argparse

import logging
import logzero
from logzero import logger
import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Cleaner and tokenizer of raw text stored as json file")
    parser.add_argument('-w', '--wordcloud_per_restaurant', default=False, type=bool, help="save wordclouds per restaurant")
    parser.add_argument('-t', '--tfidf_per_restaurant', default=False, type=bool, help='save tfidf matrix per restaurant')
    parser.add_argument('-e', '--embedding_technique', default="all", type=str, help='chooses embedding technique')
    args = parser.parse_args()

    # Check input args
    embedding_techniques = []
    if args.embedding_technique not in ['word2vec', 'lsi', 'fasttext', 'all']:
        logger.error(f"Embedding techniques supported (word2vec, lsi, fasttext), found {args.embedding_technique}")
    else:
        if args.embedding_technique == 'all':
            embedding_techniques = ['word2vec', 'lsi', 'fasttext']
        else:
            embedding_techniques = [args.embedding_technique]

    
    # Merge scraping data from different runs
    try:
        os.mkdir("./scraper/scraped_data/merged_data")
    except OSError:
        logger.warning("OSError: directory already exists")

    merge_files("./scraper/scraped_data/restaurants/", "./scraper/scraped_data/merged_data/merged_restaurants.json")
    merge_files("./scraper/scraped_data/reviews/", "./scraper/scraped_data/merged_data/merged_reviews.json")
    merge_files("./scraper/scraped_data/users/", "./scraper/scraped_data/merged_data/merged_users.json")

    # Generate balanced dataset
    reviews = pd.read_json('./scraper/scraped_data/merged_data/merged_reviews.json', lines=True)
    by_rating = reviews.groupby(by=['rating']).count()
    min_count = min(by_rating['review_id'])
    balanced_reviews = reviews.groupby("rating").sample(n=min_count, random_state=0)
    balanced_reviews.set_index(['review_id'], inplace=True)
    balanced_reviews.to_csv('./scraper/scraped_data/merged_data/balanced_reviews.csv', sep='#', index_label='review_id')
    
    # Clean balanced dataset of reviews
    cleaner = Cleaner(debug=1, early_stop=None, assets_directory= './cleaner/assets/')
    cleaner.set_file('./scraper/scraped_data/merged_data/balanced_reviews.csv')
    cleaner.preprocessing(ngram=2)
    cleaner.save_tokenized_corpus('./cleaner/cleaned_data/', 'tokenized_corpus.json')

    if args.wordcloud_per_restaurant:
        cleaner.save_files('./cleaner/cleaned_data/restaurant_wordclouds/', save_wordcloud, mask_path='./cleaner/assets/capgemini.jpg')
    if args.tfidf_per_restaurant:
        cleaner.save_files('./cleaner/cleaned_data/restaurant_word_frequencies/', save_tfidf)

    cleaner.save_sparse_matrix('./cleaner/cleaned_data/restaurant_tfidf_sparse.npz', 
                               './cleaner/cleaned_data/restaurant_tfidf_sparse_review_ids.csv',
                               './cleaner/cleaned_data/restaurant_tfidf_sparse_colnames.csv',
                               './cleaner/cleaned_data/restaurant_tfidf_sparse.txt')

    # Embed balanced dataset of reviews
    embedder = Embedder()

    if "lsi" in embedding_techniques:    
        embedder.embed("lsi", filepath='./cleaner/cleaned_data/restaurant_tfidf_sparse.npz',
                                review_id_fp='./cleaner/cleaned_data/restaurant_tfidf_sparse_review_ids.csv',
                                colnames_fp='./cleaner/cleaned_data/restaurant_tfidf_sparse_colnames.csv')
        embedder.write_files("./embedder/embedded_data/")

    if "word2vec" in embedding_techniques: 
        embedder.embed('word2vec', filepath='./cleaner/cleaned_data/tokenized_corpus.json')
        embedder.write_files('./embedder/embedded_data/')

    if "fastText" in embedding_techniques:
        embedder.embed('fastText', filepath='./cleaner/cleaned_data/tokenized_corpus.json')
        embedder.write_files('./embedder/embedded_data/')