import tensorflow as tf
from tensorforce.core.networks import Dense, Conv2d, Flatten, Network, Layer

# class Concat(Layer):
#     def __init__(self, scope='concatenate', summary_labels=()):
#         super(Concat, self).__init__(scope=scope, summary_labels=summary_labels)
#
#     def tf_apply(self, x, update):
#         return tf.concat(x, axis=-1)
#
#
# class MaxPooling(Layer):
#     def __init__(self, pool_size, strides, scope='max_pooling', summary_labels=()):
#         self.pool_size = pool_size
#         self.strides = strides
#         super(MaxPooling, self).__init__(scope=scope, summary_labels=summary_labels)
#
#     def tf_apply(self, x, update):
#         return tf.layers.max_pooling2d(x, self.pool_size, self.strides)
#
#
# class CustomNetwork(Network):
#     def __init__(self, scope='PPONet', summary_labels=(), seed=0):
#         super(CustomNetwork, self).__init__(scope=scope, summary_labels=summary_labels)
#         initializer = tf.glorot_normal_initializer(seed=seed)
#
#         self.dence_inp1 = Dense(size=32, weights=initializer, scope='dense_inp')
#         self.dence_inp2 = Dense(size=32, weights=initializer, scope='dense_inp')
#
#         self.conv1 = Conv2d(32, 2, activation='relu', scope='conv_inp', padding='VALID')
#         self.max_pool1 = MaxPooling(2, 2, scope='conv_inp')
#         self.conv2 = Conv2d(64, 3, activation='relu', scope='conv_inp', padding='VALID')
#         self.conv3 = Conv2d(64, 3, activation='relu', scope='conv_inp', padding='VALID')
#         self.conv_inp_fl = Flatten(scope='conv_inp')
#
#         self.inp_concat = Concat(scope='inp')
#         self.fc1 = Dense(128, weights=initializer, scope='dence_1')
#         self.fc2 = Dense(128, weights=initializer, scope='dence_2')
#
#     def tf_apply(self, x, internals, update, return_internals=False):
#         board = x['board']
#         state = x['state']
#
#         inp1 = self.dence_inp1.apply(x=state, update=update)
#         inp1 = self.dence_inp2.apply(x=inp1, update=update)
#
#         inp2 = self.conv1.apply(x=board, update=update)
#         inp2 = self.max_pool1.apply(x=inp2, update=update)
#         inp2 = self.conv2.apply(x=inp2, update=update)
#         inp2 = self.conv3.apply(x=inp2, update=update)
#         inp2 = self.conv_inp_fl.apply(x=inp2, update=update)
#
#         inp = self.inp_concat.apply(x=[inp1, inp2], update=update)
#         out = self.fc1.apply(x=inp, update=update)
#         out = self.fc2.apply(x=out, update=update)
#
#         if return_internals:
#             return out, list()
#         return out

class PommNetwork(Network):
    def tf_apply(self, x, internals, update, return_internals=False):
        fc = self.create_network(x)
        # TODO maybe {} is not ok
        if return_internals:
            return fc, {}
        return fc

    @staticmethod
    def create_network(x):
        board = x['board']
        state = x['state']

        with tf.variable_scope('PPONet'):
            with tf.variable_scope("DenseModel"):
                dence1 = tf.layers.dense(state, 32, activation=tf.nn.relu)
                fc_inp1 = tf.layers.dense(dence1, 32, activation=tf.nn.relu)

            with tf.variable_scope("ConvNetModel"):
                conv1 = tf.layers.conv2d(board, 32, 2, activation=tf.nn.relu)
                conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

                conv2 = tf.layers.conv2d(conv1, 64, 2, activation=tf.nn.relu)
                conv2 = tf.layers.conv2d(conv2, 64, 3, activation=tf.nn.relu)
                conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
                fc_inp2 = tf.layers.flatten(conv2)

            fc_inp = tf.concat([fc_inp1, fc_inp2], -1)
            fc = tf.layers.dense(fc_inp, 128, activation=tf.nn.relu)
            fc = tf.layers.dense(fc, 128, activation=tf.nn.relu)
        return fc