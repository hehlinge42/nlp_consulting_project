import argparse
from embedder import Embedder
from model import RatingPredictor

from logzero import logger
import logzero
import logging 
from datetime import datetime
import os

def embed():

    start = datetime.now()
    logger.critical(f"Launching embedder at {start}")

    embedder = Embedder()
    embedder.embed("lsi", filepath='../../cleaner/cleaned_data/restaurant_tfidf_sparse.npz',
                            review_id_fp='../../cleaner/cleaned_data/restaurant_tfidf_sparse_review_ids.csv',
                            colnames_fp='../../cleaner/cleaned_data/restaurant_tfidf_sparse_colnames.csv')
    embedder.write_files("../embedded_data/")

    embedder.embed('word2vec', filepath='../../cleaner/cleaned_data/tokenized_corpus.json')
    embedder.write_files('../embedded_data/')

    embedder.embed('fastText', filepath='../../cleaner/cleaned_data/tokenized_corpus.json')
    embedder.write_files('../embedded_data/')

    end = datetime.now()
    logger.critical(f"Ending embedder at {end}")
    logger.critical(f"Embedder took {end - start}")

def classify():

    rating_predictor = RatingPredictor()
    # embedders = ['lsi', 'word2vec', 'fasttext']
    embedders = ['spark_lsi']
    for embedder in embedders:
        rating_predictor.set_Xy_train(input=embedder)
        rating_predictor.generate_model()
        print(str(rating_predictor))
        rating_predictor.train_test_model(epochs=30, batch_size=32, validation_split=0.2, early_stopping_monitor=None)
        rating_predictor.save_model(os.path.join('..', 'trained_models'), filename=embedder + '.h5')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Embeds tokenized reviews and feeds them to a Neural Network")
    parser.add_argument('-e', '--embed', nargs="*", type=bool, default=False, help='Embeds tokenized reviews')
    parser.add_argument('-c', '--classify', type=bool, default=False, help="Predict rating from embedded reviews")
    args = parser.parse_args()

    if args.embed:
        embed()
    if args.classify:
        classify()