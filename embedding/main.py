import argparse
from embedder import Embedder

from logzero import logger
import logzero
import logging 
from datetime import datetime


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="Cleaner and tokenizer of raw text stored as json file")
    # parser.add_argument('-f', '--files', nargs="*", type=str, help='path to the files to be cleaned')
    # parser.add_argument('-d', '--debug', help="prints intermediary logs", action="store_true")
    # parser.add_argument('-s', '--early_stop', type=int, default=-1, help='Caps the number of reviews to be processed')
    # args = parser.parse_args()

    start = datetime.now()
    logger.critical(f"Launching embedder at {start}")
    embedder = Embedder()
    embedder.embed("lsi")
    embedder.write_files("lsi_embed/")
    end = datetime.now()
    logger.critical(f"Ending embedder at {end}")
    logger.critical(f"Embedder took {end - start}")

    # embedder = Embedder(filepath='../cleaner/cleaned_data/tokenized_corpus.json')
    # embedder.embed('word2vec')
    # embedder.write_files('word2vec_embed/')