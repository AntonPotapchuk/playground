import tensorflow as tf
from keras import Model
from keras.layers import Convolution2D, BatchNormalization, Activation, Add, Dense
from keras.models import load_model
from tensorforce.core.networks import Network
from keras.engine import Input

class PommNetwork(Network):
    def tf_apply(self, x, internals, update, return_internals=False):
        fc = self.create_network(x)
        # TODO maybe {} is not ok
        if return_internals:
            return fc, {}
        return fc

    @staticmethod
    def create_network(board):
        inp = Input(tensor = board['board'])
        x = Dense(8)(inp)
        x = Activation('relu')(x)
        x = Dense(8)(x)
        x = Activation('relu')(x)
        out = Dense(6)(x)
        out = Activation('softmax')(out)
        model = Model(inputs=inp, outputs=out)
        model.load_weights('./dqn/model/ddgp_dense_8_2/model.h4')

        return out