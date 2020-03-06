import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Bidirectional,GRU,SpatialDropout1D,GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate
from util import ID_TO_CLASS

import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.ERROR)


class MyBasicAttentiveBiGRU(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()

        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        ### TODO(Students) START
        self.bigru_layer = Bidirectional(GRU(hidden_size, return_sequences=True))
        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START
        M = tf.math.tanh(rnn_outputs) #Equation 9
        alpha = tf.nn.softmax(tf.tensordot(M,self.omegas, axes=1), axis=1) #Equation 10
        r = tf.reduce_sum(tf.multiply(rnn_outputs,alpha),1) #Equation 11 and weighted sum
        ### TODO(Students) END
        return r

    def call(self, inputs, pos_inputs, training):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        ### TODO(Students) START
        #feature ---> batch_size x 2*embed_dim
        total_features = word_embed#tf.concat([word_embed, pos_embed],2)
        sequence_mask = tf.cast(inputs != 0, tf.float32)
        timesteps = self.bigru_layer(total_features, mask=sequence_mask)
        h_star = tf.math.tanh(self.attn(timesteps)) #Equation 12
        ### TODO(Students) END
        logits = self.decoder(h_star)
        return {'logits': logits}


class MyAdvancedModel(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 80, training: bool = False):
        super(MyAdvancedModel, self).__init__()
        ### TODO(Students) START
        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))
        self.spatial_dropout_layer = SpatialDropout1D(0.35)
        self.bigru_layer = Bidirectional(GRU(hidden_size, return_sequences=True, return_state=True, dropout=0.1, recurrent_dropout=0.1))
        self.avg_layer = GlobalAveragePooling1D()
        self.max_layer = GlobalMaxPooling1D()

    ### TODO(Students END

    def call(self, inputs, pos_inputs, training):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)

        ### TODO(Students) START
        # feature ---> batch_size x 2*embed_dim
        ### TODO(Students) START
        features = self.spatial_dropout_layer(word_embed)
        time_steps, final_state, final_cell_state = self.bigru_layer(features)
        avg_pool = self.avg_layer(time_steps)
        max_pool = self.max_layer(time_steps)
        conc = concatenate([max_pool, avg_pool, final_state])
        logits = self.decoder(conc)
        return {'logits': logits}

