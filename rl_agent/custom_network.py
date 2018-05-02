from tensorflow-gp

from tensorforce.core.networks import Network

class CustomNetwork(Network):
    def tf_apply(self, x, internals, update, return_internals=False):
        
        raise NotImplementedError