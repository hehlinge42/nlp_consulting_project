import argparse
from embedder import Embedder
from model import RatingPredictor

from logzero import logger
import logzero
import logging 
from datetime import datetime
import os

def embed(cleaned_data_dir, embedded_data_dir):
    # ./cleaner/cleaned_data
    # './embedder/embedded_data/'

    start = datetime.now()
    logger.critical(f'Launching embedder at {start}')

    embedder = Embedder()
    
    embedder.embed('lsi', filepath=os.path.join(cleaned_data_dir, 'restaurant_tfidf_sparse.npz'),
                            review_id_fp=os.path.join(cleaned_data_dir, 'restaurant_tfidf_sparse_review_ids.csv'),
                            colnames_fp=os.path.join(cleaned_data_dir, 'restaurant_tfidf_sparse_colnames.csv'))
    embedder.write_files(embedded_data_dir)

    embedder.embed('word2vec', filepath=os.path.join(cleaned_data_dir, 'tokenized_corpus.json'))
    embedder.write_files(embedded_data_dir)

    embedder.embed('fastText', filepath=os.path.join(cleaned_data_dir, 'tokenized_corpus.json'))
    embedder.write_files(embedded_data_dir)

    end = datetime.now()
    logger.critical(f'Ending embedder at {end}')
    logger.critical(f'Embedder took {end - start}')

def classify(reviews_fp, embed_type, best_params_fp, trained_models_dir):
    
    rating_predictor = RatingPredictor(reviews_fp)
 
    rating_predictor.set_Xy_train(best_params_fp, input=embed_type)
    rating_predictor.generate_model()
    print(str(rating_predictor))
    rating_predictor.train_test_model(validation_split=0.2, early_stopping_monitor=None)
    rating_predictor.save_model(trained_models_dir, filename=embedder + '.h5')


if __name__ == '__main__':

    path_list = os.getcwd().split(os.sep)
    target_index = path_list.index('nlp_consulting_project')
    running_dir = os.path.join('.', path_list[target_index + 1])
    path_list = path_list[:target_index + 1]
    os.chdir(os.path.join(os.sep, *path_list))

    parser = argparse.ArgumentParser(description='Embeds tokenized reviews and feeds them to a Neural Network')
    parser.add_argument('-e', '--embed', nargs='*', type=bool, default=False, help='Embeds tokenized reviews')
    parser.add_argument('-c', '--cleaned_data_dir', type=str, help='Directory to find cleaned data')
    parser.add_argument('-d', '--embedded_data_dir', type=str, help='Directory to save embedded data')
    parser.add_argument('-m', '--model', type=bool, default=False, help='Predict rating from embedded reviews')
    parser.add_argument('-t', '--trained_models_dir', type=str, help='Directory to save trained classification models')
    args = parser.parse_args()

    if args.embed:
        embed(args.cleaned_data_dir, args,embedded_data_dir)
    if args.model:
        merged_reviews_fp = './scraper/scraped_data/merged_data/balanced_reviews.csv'
        best_params_fp = './embedder/trained_models/' + str(embed_type) + '_params.json'
        embed_type = 'spark_lsi'
        classify(merged_reviews_fp, embed_type, best_params_fp, args.trained_models_dir)