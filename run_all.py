
import logging
import logzero
from logzero import logger

from scraper.merger import merge_files

from cleaner.src.cleaner import Cleaner
from embedder.src.embedder import Embedder
from cleaner.src.helpers import save_wordcloud, save_tfidf
from embedder.src.model import RatingPredictor

import json
import os
import argparse
import pandas as pd
import PySimpleGUI as sg

def run_all(settings):

    path_list = os.getcwd().split(os.sep)
    target_index = path_list.index('nlp_consulting_project')
    path_list = path_list[:target_index + 1]
    if os.name=='nt':
        path_list[0] += '/'
    os.chdir(os.path.join(os.sep, *path_list))

    # Check input args
    embedding_techniques = []
    user_embedding = settings['embedding_technique']
    if user_embedding not in ['word2vec', 'lsi', 'fasttext', 'all']:
    # if args.embedding_technique not in ['word2vec', 'lsi', 'fasttext', 'all']:
        logger.error(f"Embedding techniques supported (word2vec, lsi, fasttext), found {user_embedding}")
    else:
        if user_embedding == 'all':
            embedding_techniques = ['word2vec', 'lsi', 'fasttext']
        else:
            embedding_techniques = [user_embedding]

    
    # Merge scraping data from different runs
    scraped_data_dir = os.path.join('scraper', 'scraped_data')
    cleaned_data_dir = os.path.join('cleaner', 'cleaned_data')
    embedded_data_dir = os.path.join('embedder', 'embedded_data')

    try:
        os.mkdir(os.path.join(scraped_data_dir, 'merged_data'))
    except OSError:
        logger.warning("OSError: directory already exists")

    merge_files(os.path.join(scraped_data_dir, 'restaurants'), os.path.join(scraped_data_dir, 'merged_data', 'merged_restaurants.json'))
    merge_files(os.path.join(scraped_data_dir, 'reviews'), os.path.join(scraped_data_dir, 'merged_data', 'merged_reviews.json'))
    merge_files(os.path.join(scraped_data_dir, 'users'), os.path.join(scraped_data_dir, 'merged_data', 'merged_users.json'))
    merge_files(os.path.join(scraped_data_dir, 'bulk_restaurants'), os.path.join(scraped_data_dir, 'merged_data', 'merged_bulk_restaurants.json'))

    # Generate balanced dataset
    path = os.path.join(os.getcwd(), os.path.join(scraped_data_dir, 'merged_data', 'merged_reviews.json'))
    reviews = pd.read_json(path, lines=True)

    by_rating = reviews.groupby(by=['rating']).count()
    min_count = min(by_rating['review_id'])
    balanced_reviews = (reviews.groupby("rating")).sample(n=min_count, random_state=0)
    balanced_reviews.set_index(['review_id'], inplace=True)
    balanced_reviews.to_csv(os.path.join(scraped_data_dir, 'merged_data', 'balanced_reviews.csv'), sep='#', index_label='review_id')
    
    # Clean balanced dataset of reviews
    cleaner = Cleaner(debug=1, early_stop=None, assets_directory= os.path.join('cleaner', 'assets'))
    cleaner.set_file(os.path.join(scraped_data_dir, 'merged_data', 'balanced_reviews.csv'))
    cleaner.preprocessing(ngram=2)
    cleaner.save_tokenized_corpus(cleaned_data_dir, 'tokenized_corpus.json')

    if int(settings['wordclouds']) == 1:
        cleaner.save_files(os.path.join(cleaned_data_dir, 'restaurants_wordclouds'), save_wordcloud, mask_path=os.path.join('cleaner', 'assets', 'capgemini.jpg'))
    if int(settings['tfidf']) == 1:
        cleaner.save_files(os.path.join(cleaned_data_dir, 'restaurants_tfidf'), save_tfidf)

    cleaner.save_sparse_matrix(os.path.join(cleaned_data_dir, 'restaurant_tfidf_sparse.npz'), 
                               os.path.join(cleaned_data_dir, 'restaurant_tfidf_sparse_review_ids.csv'),
                               os.path.join(cleaned_data_dir, 'restaurant_tfidf_sparse_colnames.csv'),
                               os.path.join(cleaned_data_dir, 'restaurant_tfidf_sparse.txt'))


    logger.info(' > Embedding reviews.')
    embedder = Embedder()

    if "lsi" in embedding_techniques:    
        embedder.embed("lsi", filepath=os.path.join(cleaned_data_dir, 'restaurant_tfidf_sparse.npz'),
                            review_id_fp=os.path.join(cleaned_data_dir, 'restaurant_tfidf_sparse_review_ids.csv'),
                            colnames_fp=os.path.join(cleaned_data_dir, 'restaurant_tfidf_sparse_colnames.csv'))
        embedder.write_files(embedded_data_dir)

    if "word2vec" in embedding_techniques: 
        embedder.embed('word2vec', filepath=os.path.join(cleaned_data_dir, 'tokenized_corpus.json'), review_id_fp=None, colnames_fp=None)
        embedder.write_files(embedded_data_dir)

    if "fasttext" in embedding_techniques:
        embedder.embed('fastText', filepath=os.path.join(cleaned_data_dir, 'tokenized_corpus.json'), review_id_fp=None, colnames_fp=None)
        embedder.write_files(embedded_data_dir)

    if "fastText" in embedding_techniques:
        embedder.embed('fastText', filepath=os.path.join(cleaned_data_dir, 'tokenized_corpus.json'))
        embedder.write_files(embedded_data_dir)

    # Classify
    logger.info(' > Classifying reviews.')

    reviews_fp = os.path.join('scraper', 'scraped_data', 'merged_data', 'balanced_reviews.csv')
    embed_type = 'spark_lsi'
    best_params_fp = os.path.join('embedder', 'trained_models', str(embed_type) + '_params.json')

    rating_predictor = RatingPredictor(reviews_fp) 
    rating_predictor.set_Xy_train(best_params_fp, input=embed_type)
    rating_predictor.generate_model()
    rating_predictor.train_test_model(validation_split=0.2, early_stopping_monitor=None)
    trained_models_dir = os.path.join('embedder', 'trained_models')
    rating_predictor.save_model(trained_models_dir, filename=embed_type + '.h5')