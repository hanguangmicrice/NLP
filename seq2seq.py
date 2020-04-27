import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model

class Encoder(Model):
    def __init__(self, voc_size,embeding_dim,enc_units,batch_size):
        super(Encoder,self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = layers.Embedding(voc_size,embeding_dim)
        self.lstm = layers.LSTM(self.enc_units,
                                return_sequences=True,
                                return_state=True)
    def call(self,x,hidden):
        x = self.embedding(x)
        output, state = self.lstm(x,hidden)
        return output, state
    def initialize_hidden_state(self):
        tf.ones((self.batch_size,self.enc_units))

class Attention(Model):
    def __init__(self):
        super().__init__()