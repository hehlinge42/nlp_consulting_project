
import numpy as np
import json

from logzero import logger
import logzero
import pandas as pd
import logging 
import os

from sklearn.decomposition import TruncatedSVD

# NLP Libraries
import gensim
import fasttext
from gensim.test.utils import get_tmpfile
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class Embedder():

    """ """

    def __init__(self, filepath='../cleaner/cleaned_data/restaurants_tfidf.csv'):
        self.filepath = filepath # document term matrix directory
        self.embedded_per_restaurant = {}
        self.tokenized_per_restaurant = {}
        logzero.loglevel(logging.DEBUG)
        

    def get_best_nb_component(self, var_threshold=0.5, max_components=100):
        # Set initial variance explained so far
        total_variance = 0.0
        n_components = 0
        
        large_svd = TruncatedSVD(n_components=self.document_term_matrix.shape[1]-1)
        large_lsa = large_svd.fit_transform(self.document_term_matrix)
        # print("VARIANCE RATIO", large_svd.explained_variance_ratio_)
        # For the explained variance of each feature:
        for explained_variance in large_svd.explained_variance_ratio_:
            total_variance += explained_variance
            n_components += 1
        
            if total_variance >= var_threshold:
                break
                
        # Return the number of components
        return n_components

    def embed(self, embedding_type, **kwargs):
        
        self.type = embedding_type
        if self.type == 'lsi':
            self.document_term_matrix = pd.read_csv(self.filepath, index_col='review_id')
            self.best_nb_component = self.get_best_nb_component()
            print("BEST COMPONENT NB", self.best_nb_component)
            self.lsi(self.best_nb_component)

        elif self.type == 'word2vec':
            with open(self.filepath) as json_file:
                document = json.load(json_file)
                self.document_term_matrix = list(document.values())
                self.review_id = list(document.keys())
            self.word2vec()

        elif self.type == 'fastText':
            self.fast_text()
        
    def lsi(self, n_components=5):
        """ """

        svd = TruncatedSVD(n_components=n_components)
        self.embedded_document = pd.DataFrame(svd.fit_transform(self.document_term_matrix))
        self.embedded_document.index = self.document_term_matrix.index

    def word2vec(self, size=self.document_term_matrix.shape[0], window=3, min_count=5, workers=4, seed=1, iter=50):
        """ """

        model = gensim.models.Word2Vec(size=size, window=window, min_count=min_count, workers=workers, seed=seed, iter=iter)
        model.build_vocab(self.document_term_matrix)
        model.train(self.document_term_matrix, total_examples=model.corpus_count, epochs=model.iter)
        embedding_matrix = {}
        for word in model.wv.vocab.keys(): # words as columns in df
            embedding_matrix[word] = list(model.wv[word])
        self.embedded_document = pd.DataFrame(embedding_matrix, index=self.review_id)

    def fast_text(self, model='skipgram'):
        """ """
        model = fasttext.train_unsupervised(self.document_term_matrix, model=model)

    def write_files(self, directory, restaurant_ids='all'):
        """ Saves embedded files for corpora """
        
        # logger.warn(f' > SAVING {self.type} FILES ')

        try:
            os.mkdir(directory)
        except OSError:
            logger.warn("OSError: directory already exists")

        self.embedded_document.to_csv(directory + str(self.type) + '.csv', index_label='review_id')
