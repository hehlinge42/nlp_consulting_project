
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout

from sklearn.model_selection import StratifiedShuffleSplit

from logzero import logger
import logzero
import logging
import os
import numpy as np

class RatingPredictor(tf.keras.Model):

    def __init__(self, path_to_reviews='../../scraper/scraped_data/merged_data/balanced_reviews.csv'):
        
        super(RatingPredictor, self).__init__()
        self.path_to_reviews = path_to_reviews
        self.reviews = None


    def __str__(self):
        return str(self.model.summary())

    
    def set_Xy_train(self, input='lsi'):

        if self.reviews is None:
            logger.info(f"Reads review file {self.path_to_reviews}")
            reviews = pd.read_csv(self.path_to_reviews, sep='#', index_col=['review_id'])
            self.reviews = reviews['rating']

        root_path = os.path.join('..', 'embedded_data')
        review_id_path = os.path.join('..', '..', 'cleaner', 'cleaned_data')
        if input == 'lsi':
            X = pd.read_csv(os.path.join(root_path, 'lsi.csv'), index_col=['review_id'])
        elif input == 'word2vec':
            X = pd.read_csv(os.path.join(root_path, 'word2vec.csv'), index_col=['review_id'])
        elif input == 'fasttext':
            X = pd.read_csv(os.path.join(root_path, 'fastText.csv'), index_col=['review_id'])
        elif input == 'spark_lsi':
            review_ids = pd.read_csv(os.path.join(review_id_path, 'restaurant_tfidf_sparse_review_ids.csv'), index_col=['review_id'])
            X = pd.read_csv(os.path.join(root_path, 'spark_lsi.csv'))
            X['review_id'] = review_ids.index.values
            X.set_index(['review_id'], inplace=True)
        else:
            logger.critical(f"Tried set_Xy_train with input {input}")
            raise NotImplementedError("Input must be 'lsi', 'word2vec' or 'fasttext'")
           
        Xy = X.merge(self.reviews, left_index=True, right_index=True)
        Xy['rating'] -= 1
        y = Xy['rating']

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        for train_index, test_index in sss.split(X, y):
            self.X_train, self.X_test = X.iloc[train_index], X.iloc[test_index]
            self.y_train, self.y_test = y.iloc[train_index], y.iloc[test_index]
        
        self.output_size = len(Xy['rating'].unique())
        self.y_train = tf.keras.utils.to_categorical(self.y_train, num_classes=None)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, num_classes=None)
        self.input_size = len(list(self.X_train.columns))

        self.X_train.to_csv('../embedded_data/X_train_' + input + '.csv')
        self.X_test.to_csv('../embedded_data/X_test_' + input + '.csv')
        with open('../embedded_data/y_train_' + input + '.npy', 'wb+') as f:
            np.save(f, self.y_train)
        with open('../embedded_data/y_test_' + input + '.npy', 'wb+') as f:
            np.save(f, self.y_test)


    def generate_model(self):

        self.model = tf.keras.models.Sequential()
        self.model.add(Input(shape=(self.input_size,)))
        self.model.add(Dense(256, activation=tf.nn.relu))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation=tf.nn.relu))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation=tf.nn.relu))
        self.model.add(Dense(self.output_size, activation=tf.nn.softmax))
        self.model.compile(loss='categorical_crossentropy', metrics='accuracy', optimizer='adam')
        

    def train_test_model(self, epochs, batch_size=32, validation_split=0.2, 
                         early_stopping_monitor=None, **kwargs):
        """ 
        Creates a given model, fits the data and returns trained model and 
        associated metrics.
        Params: 
        -model_creation (callable): model to create
        -X_train, y_train, X_test, y_test: train and test sets
        -batch_size (int): batch size used in training the model
        -early_stopping_monitor: EarlyStopping Keras object
        -**kwargs (dict): input parameters used to create the model
        Returns: Trained model, history, and accuracy on train and test sets
        """
        
        history = self.model.fit(self.X_train, self.y_train, epochs=epochs, 
                    batch_size=batch_size, validation_split=validation_split, 
                    callbacks=early_stopping_monitor)
        print('Evaluating train accuracy:')
        _, acc_train = self.model.evaluate(self.X_train, self.y_train)
        print('Evaluating test accuracy:')
        _, acc_test = self.model.evaluate(self.X_test, self.y_test)
            
        return history, acc_train, acc_test

        
    def save_model(self, directory, filename):

        try:
            os.mkdir(directory)
        except OSError:
            logger.warn("OSError: directory already exists")

        self.model.save(os.path.join(directory, filename))
    
    