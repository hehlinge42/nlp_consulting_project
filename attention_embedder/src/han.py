import tensorflow as tf
from attention import Attention

class HierarchicalAttentionNetwork(tf.keras.Model):
    """Hierarchical Attention Network implementation.

    Reference :
    * Hierarchical Attention Networks for Document Classification : https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf

    """
    def __init__(self, vocab_size, embedding_dim, gru_units, attention_units, 
                 classifier_units,
                 pretrained_weights=None, mask_zero=True,
                 dropout_embedding=0.0):
        """Hierarchical Attention Network class constructor.

        """
        super(HierarchicalAttentionNetwork, self).__init__()
        
        if pretrained_weights is not None:
            initializer = tf.keras.initializers.Constant(pretrained_weights)
        else:
            initializer = "uniform"

        # Regularisation
        self.dropout_embedding = dropout_embedding
        # self.recurrent_dropout = recurrent_dropout

        # Main Layers
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, 
            embedding_dim, 
            embeddings_initializer=initializer,
            trainable=False
            # mask_zero=mask_zero
        )
        self.WordGRU = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                units=gru_units,
                activation="tanh",
                return_sequences=True,
                # recurrent_dropout=self.recurrent_dropout
            ), 
            merge_mode='concat',
        )
        self.WordAttention = Attention(units=attention_units)

        self.SentenceGRU = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                units=gru_units,
                activation="tanh",
                return_sequences=True,
                # recurrent_dropout=self.recurrent_dropout
            ), 
            merge_mode='concat',
        )
        self.SentenceAttention = Attention(units=attention_units)

        self.fc = tf.keras.layers.Dense(units=classifier_units, activation=tf.keras.activations.softmax) 
                                        # activity_regularizer=callable(penalty))

    def call(self, x):
        """Model forward method.
        """
        sentences_vectors, _ = self.word_to_sentence_encoder(x)
        document_vector, _ = self.sentence_to_document_encoder(sentences_vectors)
        return self.fc(document_vector)

    def word_to_sentence_encoder(self, x):
        """Given words from each sentences, encode the contextual representation of 
        the words from the sentence with Bidirectional GRU and Attention, and output 
        a sentence_vector.
        """
        x = self.embedding(x)
        mask = self.embedding.compute_mask(x)
        if self.dropout_embedding > 0.0:
          x = tf.keras.layers.Dropout(self.dropout_embedding)(x)
        x = tf.keras.layers.TimeDistributed(self.WordGRU)(x)
        context_vector, attention_weights = self.WordAttention(x)

        return context_vector, attention_weights
    
    def sentence_to_document_encoder(self, sentences_vectors):
        """Given sentences from each review, encode the contextual representation of 
        the sentences with Bidirectional GRU and Attention, and output 
        a document vector.
        """
        # sentence encoder (using bidirectionnal GRU)
        sentences_vectors = self.SentenceGRU(sentences_vectors)        
        # document vector  (using attention at sentence level)
        document_vector, attention_weights = self.SentenceAttention(sentences_vectors)
        
        return document_vector, attention_weights