import numpy as np
import json

from logzero import logger
import logzero
import pandas as pd
import logging 
import os

from sklearn.decomposition import TruncatedSVD

from scipy.sparse import load_npz
from scipy.sparse.linalg import svds

# NLP Libraries
import gensim
import fasttext
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class Embedder():

    """ """

    def __init__(self, filepath='../cleaner/cleaned_data/restaurant_tfidf_sparse.npz'):
        self.filepath = filepath # document term matrix directory
        self.embedded_per_restaurant = {}
        self.tokenized_per_restaurant = {}
        logzero.loglevel(logging.DEBUG)
        

    def get_best_nb_component(self, var_threshold=0.5, max_components=1):

        total_variance = 0.0
        n_components = 0
        max_components = min(max_components, self.document_term_matrix.shape[1]-1)

        logger.info(f"Starting get_best_nb_component with {max_components} components")
        u, s, vt = svds(self.document_term_sparse_matrix, max_components)
        s = np.diag(s)
        logger.info(f"shape of u {u.shape}")
        logger.info(f"shape of s {s.shape}")
        logger.info(f"shape of vt {vt.shape}")
        logger.info("Starting recomposition of matrix")
        compressed_matrix = np.dot(u, np.dot(s, vt))
        logger.critical(f"compressed matrix of shape {compressed_matrix.shape}")
        # large_svd = TruncatedSVD(n_components=max_components)
        # large_lsa = large_svd.fit_transform(self.document_term_matrix)
        logger.info("Ending get_best_nb_component")

        for explained_variance in large_svd.explained_variance_ratio_:
            total_variance += explained_variance
            n_components += 1
        
            if total_variance >= var_threshold:
                break
                
        # Return the number of components
        return n_components

    def embed(self, embedding_type, review_id_fp='../cleaner/cleaned_data/restaurant_tfidf_sparse_review_ids.csv',
              colnames_fp='../cleaner/cleaned_data/restaurant_tfidf_sparse_colnames.csv', **kwargs):
        
        self.type = embedding_type
        if self.type == 'lsi':

            self.document_term_sparse_matrix = load_npz(self.filepath)
            dense_matrix = self.document_term_sparse_matrix.todense()
            review_id = pd.read_csv(review_id_fp, index_col='review_id')
            colnames = pd.read_csv(colnames_fp)
            self.document_term_matrix = pd.DataFrame(dense_matrix, index=review_id.index.values, columns=colnames['colnames'])

            self.best_nb_component = self.get_best_nb_component()
            logger.info(f"BEST COMPONENT is {self.best_nb_component}")
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


    def word2vec(self, window=3, min_count=5, workers=4, seed=1, iter=50):
        """ """
        
        size = len(self.document_term_matrix)
        model = gensim.models.Word2Vec(size=size, window=window, min_count=min_count, workers=workers, seed=seed, iter=iter)
        model.build_vocab(self.document_term_matrix)
        model.train(self.document_term_matrix, total_examples=model.corpus_count, epochs=model.iter)
        embedding_matrix = {}
        for word in model.wv.vocab.keys(): # words as columns in df
            embedding_matrix[word] = list(model.wv[word])
        self.embedded_document = pd.DataFrame(embedding_matrix, index=self.review_id)

    def fast_text(self, model='skipgram'):
        """ """

        model = FastText(vector_size=4, window=3, min_count=1)  # instantiate
        model.build_vocab(sentences=self.document_term_matrix)
        model.train(sentences=self.document_term_matrix, total_examples=len(self.document_term_matrix), epochs=10)
        
        # model = fasttext.train_unsupervised(self.document_term_matrix, model=model)

    def write_files(self, directory, restaurant_ids='all'):
        """ Saves embedded files for corpora """
        
        # logger.warn(f' > SAVING {self.type} FILES ')

        try:
            os.mkdir(directory)
        except OSError:
            logger.warn("OSError: directory already exists")

        self.embedded_document.to_csv(directory + str(self.type) + '.csv', index_label='review_id')
