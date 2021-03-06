import tensorflow as tf
import logging
import logzero
from logzero import logger

from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

from han import HierarchicalAttentionNetwork

def perform_simple_model(train_ds, test_ds, pretrained_weights):

    vocab_size = pretrained_weights.shape[0]

    simple_rating_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            vocab_size, 
            128, 
            embeddings_initializer=tf.keras.initializers.Constant(pretrained_weights),
            trainable=True
        ),
        tf.keras.layers.Dense(64),
        tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last'),
        tf.keras.layers.Dense(5)
    ])

    simple_rating_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=["accuracy"]
    )

    simple_history = simple_rating_model.fit(
        train_ds, 
        epochs=20, 
        validation_data=test_ds
    )

def perform_han_model(train_ds, test_ds, pretrained_weights):
    
    han_model_reg = HierarchicalAttentionNetwork(vocab_size=pretrained_weights.shape[0], embedding_dim=128, 
                    pretrained_weights=pretrained_weights, gru_units=32, attention_units=32, 
                    classifier_units=5) #dropout_embedding=0., recurrent_dropout=0., callable=tf.keras.regularizers.l2, penalty=1e-05)


    han_model_reg.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                        optimizer="adam", metrics=["accuracy"])

    han_history_reg = han_model_reg.fit(train_ds, epochs=5, validation_data=test_ds)