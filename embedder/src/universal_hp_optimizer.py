from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import datetime
import numpy as np
import pandas as pd
import itertools as it
import os


DEFAULT_PARAMS = {
    "epochs": hp.HParam('epochs', hp.Discrete([10])),
    'batch_size': hp.HParam('batch_size', hp.Discrete([16, 32, 64])),
    'batch_normalization': hp.HParam('batch_normalization', hp.Discrete([False], dtype=bool)),
    'dropout': hp.HParam('dropout', hp.Discrete([0])),
    'optimizer': hp.HParam('optimizer', hp.Discrete(['SGD'])),
    'early_stopping': hp.HParam('early_stopping', hp.Discrete([3]))
}

class UniversalHPOptimizer():
    """ A class to optimize a given model and a given dictionary of parameters
    to test. It uses the Tensorboard API in order to log the results.
    
    """

    def __init__(self, dict_params, create_model, log_dir="logs/fit/", print_summary=False):
        """

        Args:
            dict_params ([dict]): [Dictionary which values are lists of possible
             values for each parameter to test in the GridSearch.]
            create_model ([funct]): [Function to create the model.]
            print_summary (bool, optional): [Whether to print a summary of the model.
            ]. Defaults to False.
        """
        self.user_input = dict_params
        self.create_model = create_model
        self.print_summary=print_summary
        self.best_model = None
        self.best_accuracy = None
        self.METRIC_ACCURACY = 'accuracy'
        self.log_dir = log_dir

        self.HP_BATCH_NORMALIZATION = self.generate_hp_dict('batch_normalization', dict_params)
        self.HP_EPOCHS = self.generate_hp_dict("epochs", dict_params)
        self.HP_BATCH_SIZE = self.generate_hp_dict('batch_size', dict_params)
        self.HP_DROPOUT = self.generate_hp_dict('dropout', dict_params)
        self.HP_OPTIMIZER = self.generate_hp_dict('optimizer', dict_params)
        self.HP_EARLY_STOPPING = self.generate_hp_dict('early_stopping', dict_params)

        self.HP_NB_COLUMNS = dict_params['nb_columns']


    def generate_hp_dict(self, category_name, dict_params):
        
        if category_name not in dict_params:
            return DEFAULT_PARAMS[category_name]
        else:
            return dict_params[category_name]


    def run_all(self, x_train, y_train, x_test, y_test):
        """[summary]

        Args:
          x_train ([ndarray]): [Array containing the training images.]
          y_train ([ndarray]): [Array containing the labels of the training 
          images.]
          x_test ([ndarray]): [Array containing the training images.]
          y_test ([ndarray]): [Array containing the labels of the training 
          images.]
        """
        
        # !rm -rf ./logs/
        session_num = 1

        all_params = {}
        all_params['dropout'] = self.HP_DROPOUT.domain.values
        all_params['epochs'] = self.HP_EPOCHS.domain.values
        all_params['batch_size'] = self.HP_BATCH_SIZE.domain.values
        all_params['optimizer'] = self.HP_OPTIMIZER.domain.values
        all_params['batch_normalization'] = self.HP_BATCH_NORMALIZATION.domain.values
        all_params['early_stopping'] = self.HP_EARLY_STOPPING.domain.values
        all_params['nb_columns'] = self.HP_NB_COLUMNS.domain.values


        keys, values = zip(*all_params.items())
        combinations = [dict(zip(keys, v)) for v in it.product(*values)]

        for hparams in combinations:
            run_name = "run-%d" % session_num
            print('\n--- Starting trial: %s' % run_name)
            print({k: v for k, v in hparams.items()})
            self.run(hparams, x_train, y_train, x_test, y_test, 'logs/hparam_tuning/' + run_name)
            session_num += 1


    def train_test_model(self, hparams, x_train, y_train, x_test, y_test):
        """[summary]

        Args:
            hparams ([dict]): [A dictionary of model parameters. The available
            parameters are :
            - batch_normalization (boolean): Adds two Batch normalization layers
            before the dropout layers. 
            - optimizer (str) : Type of optimizer to use.
            - dropout (float) : The dropout rate, between 0 and 1.
            - batch_size (int): size of the batch.
            - epoch (int): number of epochs.
            ]
            x_train ([ndarray]): [Array containing the training images.]
            y_train ([ndarray]): [Array containing the labels of the training 
            images.]
            x_test ([ndarray]): [Array containing the training images.]
            y_test ([ndarray]): [Array containing the labels of the training 
            images.]

        Returns:
            [float]: [Returns the scalar test loss of the test.]
        """
        model = self.create_model(hparams, self.print_summary)
        log_dir = os.path.join(self.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        x_train = x_train.iloc[:, 0:hparams['nb_columns']]
        x_test = x_test.iloc[:, 0:hparams['nb_columns']]

        params_callback = {}
        for k, v in hparams.items():
            if k in self.user_input:
                params_callback[k] = v

        tensorboard_callback = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
                                hp.KerasCallback(log_dir, params_callback),
                                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=hparams["early_stopping"], verbose=1)]

        model.fit(x_train, y_train, batch_size=hparams["batch_size"], epochs=hparams["epochs"],
                  callbacks=[tensorboard_callback], validation_split=0.2) 
        _, accuracy = model.evaluate(x_test, y_test)

        if self.best_accuracy is None or accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = model
            self.best_params = hparams

        return accuracy

    def run(self, hparams, x_train, y_train, x_test, y_test, run_dir="."):
        """[summary]

        Args:
            hparams ([dict]): [A dictionary of model parameters. The available
            parameters are :
            - batch_normalization (boolean): Adds two Batch normalization layers
            before the dropout layers. 
            - optimizer (str) : Type of optimizer to use.
            - dropout (float) : The dropout rate, between 0 and 1.
            - batch_size (int): size of the batch.
            - epoch (int): number of epochs.
            ]
            x_train ([ndarray]): [Array containing the training images.]
            y_train ([ndarray]): [Array containing the labels of the training 
            images.]
            x_test ([ndarray]): [Array containing the training images.]
            y_test ([ndarray]): [Array containing the labels of the training 
            images.]
            run_dir (str, optional): [Path to the directory from which to run 
            the model.]. Defaults to ".".
        """

        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            accuracy = self.train_test_model(hparams, x_train, y_train, x_test, y_test)
            tf.summary.scalar(self.METRIC_ACCURACY, accuracy, step=1)

 
    def predict(self, x_test):
        """ Predicts the results for x_test with the model.

        Args:
            x_test ([ndarray]): [Array containing the training images.]

        Returns:
            [tuple]: [Returns an array of weights for predictions and an 
            array of predicted labels.]
        """
        predicted_probas = self.best_model.predict(x_test, verbose=1, max_queue_size=10)
        predicted_classes = np.argmax(predicted_probas, axis=-1)
        return predicted_probas, predicted_classes


    def get_confusion_matrix(self, y_true, x_test=None, y_pred=None, labels=None):
        """ Builds a confusion matrix for the model.

        Args:
            y_true ([ndarray]): [Array containing the true labels.]
            x_test ([ndarray]): [Array containing the training images.]. Defaults to None.
            y_pred ([ndarray]): [Array containing the weigths for the prediction.]. Defaults to None.
            labels ([list], optional): [List containing all of the unique labels.]. Defaults to None. 

        Returns:
            [DataFrame]: [Returns a pandas DataFrame containing the confusion matrix, 
            with the rows being the true labels and the columns the predicted labels.]
        """
        real_labels = ["real " + label for label in labels]
        pred_labels = ["pred " + label for label in labels]
        if y_pred is None:
            _, y_pred = self.predict(x_test)
        conf_matrix = tf.math.confusion_matrix(y_true, y_pred).numpy()
        df_conf_matrix = pd.DataFrame(conf_matrix, index=real_labels, columns=pred_labels)
        return df_conf_matrix