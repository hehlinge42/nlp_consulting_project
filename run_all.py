
from scraper.merger import merge_files
from cleaner.src.cleaner import Cleaner
from embedder.src.embedder import Embedder
from cleaner.src.helpers import save_wordcloud, save_tfidf
from embedder.src.model import RatingPredictor

import os
import argparse

import logging
import logzero
from logzero import logger
import pandas as pd

if __name__ == "__main__":

    path_list = os.getcwd().split(os.sep)
    target_index = path_list.index('nlp_consulting_project')
    path_list = path_list[:target_index + 1]
    os.chdir(os.path.join(os.sep, *path_list))

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

    # Generate balanced dataset
    reviews = pd.read_json(os.path.join(scraped_data_dir, 'merged_data', 'merged_reviews.json'), lines=True)
    by_rating = reviews.groupby(by=['rating']).count()
    min_count = min(by_rating['review_id'])
    balanced_reviews = reviews.groupby("rating").sample(n=min_count, random_state=0)
    balanced_reviews.set_index(['review_id'], inplace=True)
    balanced_reviews.to_csv(os.path.join(scraped_data_dir, 'merged_data', 'balanced_reviews.csv'), sep='#', index_label='review_id')
    
    # Clean balanced dataset of reviews
    cleaner = Cleaner(debug=1, early_stop=None, assets_directory= os.path.join('cleaner', 'assets'))
    cleaner.set_file(os.path.join(scraped_data_dir, 'merged_data', 'balanced_reviews.csv'))
    cleaner.preprocessing(ngram=2)
    cleaner.save_tokenized_corpus(cleaned_data_dir, 'tokenized_corpus.json')

    if args.wordcloud_per_restaurant:
        cleaner.save_files(os.path.join(cleaned_data_dir, 'restaurants_wordclouds'), save_wordcloud, mask_path=os.path.join('cleaner', 'assets', 'capgemini.jpg'))
    if args.tfidf_per_restaurant:
        cleaner.save_files(os.path.join(cleaned_data_dir, 'restaurants_tfidf'), save_tfidf)

    cleaner.save_sparse_matrix(os.path.join(cleaned_data_dir, 'restaurant_tfidf_sparse.npz'), 
                               os.path.join(cleaned_data_dir, 'restaurant_tfidf_sparse_review_ids.csv'),
                               os.path.join(cleaned_data_dir, 'restaurant_tfidf_sparse_colnames.csv'),
                               os.path.join(cleaned_data_dir, 'restaurant_tfidf_sparse.txt'))

    # Embed balanced dataset of reviews
    # embedder = Embedder()

    # if "lsi" in embedding_techniques:    
    #     embedder.embed("lsi", filepath=os.path.join(cleaned_data_dir, 'restaurant_tfidf_sparse.npz'),
    #                         review_id_fp=os.path.join(cleaned_data_dir, 'restaurant_tfidf_sparse_review_ids.csv'),
    #                         colnames_fp=os.path.join(cleaned_data_dir, 'restaurant_tfidf_sparse_colnames.csv'))
    #     embedder.write_files(embedded_data_dir)

    # if "word2vec" in embedding_techniques: 
    #     embedder.embed('word2vec', filepath=os.path.join(cleaned_data_dir, 'tokenized_corpus.json'))
    #     embedder.write_files(embedded_data_dir)

    # if "fastText" in embedding_techniques:
    #     embedder.embed('fastText', filepath=os.path.join(cleaned_data_dir, 'tokenized_corpus.json'))
    #     embedder.write_files(embedded_data_dir)

    # Classify
    reviews_fp = os.path.join('scraper', 'scraped_data', 'merged_data', 'balanced_reviews.csv')

    rating_predictor = RatingPredictor(reviews_fp)
    # embedders = ['lsi', 'word2vec', 'fasttext']
    embedders = ['spark_lsi']
    for embedder in embedders:
        rating_predictor.set_Xy_train(input=embedder)
        rating_predictor.generate_model()
        print(str(rating_predictor))
        rating_predictor.train_test_model(epochs=30, batch_size=32, validation_split=0.2, early_stopping_monitor=None)
        rating_predictor.save_model(trained_models_dir, filename=embedder + '.h5')
