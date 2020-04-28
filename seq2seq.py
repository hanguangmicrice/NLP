import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model

class Encoder(Model):
    def __init__(self, voc_size, embeding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = layers.Embedding(voc_size,embeding_dim)
        self.lstm = layers.LSTM(self.enc_units,
                                return_sequences=True,
                                return_state=True)
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state,_ = self.lstm(x, hidden)
        return output, state
    def initialize_hidden_state(self):
        tf.ones((self.batch_size, self.enc_units))

class Attention(tf.keras.layers.Layer):
    def __init__(self, unit):
        super(Attention).__init__()
        self.w1 = layers.Dense(unit)
        self.w2 = layers.Dense(unit)
        self.v = layers.Dense(1)
    def call(self, query, value):
        # change the hidden layer shape from (batch_size,unit_size)
        # as (batch_size,1,unit_size)
        hidden_with_axis = tf.expand_dims(value, 1)
        # before self.V the shape of tensor is (batch_size,length_sentence, self.unit)
        # after this thee shape change as (b,length_sentence,1)
        scores = self.v(tf.nn.tanh(self.w1(query)+self.w2(hidden_with_axis)))
        # use the softmax transfer the scores as probability
        # according we have the (batch_size, length_sentence,unit_size) input which
        # need to be assigned a weight within each input
        scores_weight = tf.nn.softmax(scores, axis=1)
        context_vector = scores_weight*value
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, scores_weight
class Decoder(Model):
    def __init__(self, voc_size, embed_dim, de_unit, batch_size):
        super(Decoder, self).__init__()
        self.de_unit = de_unit
        self.batch_size = batch_size
        self.embedding = layers.Embedding(voc_size, embed_dim)
        self.lstm = layers.LSTM(de_unit,
                                return_sequences=True,
                                return_state=True)
        self.att = Attention(self.de_unit)
        self.fc = layers.Dense(voc_size)
    def call(self, x,hidden, enc_output):
        context_output, weights_map = self.att(hidden,enc_output)

        # change the input as embedding dim
        x = self.embedding(x)
        # after above operation the shape of x becomes (batch_size,1,embedding_dim)
        # concat the output of attention layer and the tensor shape becomes the (batch_size, 1, embedding_dim + uint_of_encoder)
        x = tf.concat([tf.expand_dims(context_output,1),x], axis=-1)
        output,state,_ = self.lstm(x)
        # reshape the output as (batch_size*1, embedding_dim)
        output = tf.reshape(output,(-1,output.shape[2]))
        x = self.fc(output)
        return x, state




