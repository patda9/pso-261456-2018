import numpy as np
from mlp import form_network

class Individual(object):
    def __init__(self, n_layer, n_weights, structure):
        weights = np.random.randn(n_weights, 1)
        
        self.fx = 0
        self.n_layer = n_layer
        self.structure = structure
        self.weights = weights
        self.layers = form_network(self.n_layer, np.copy(self.weights), self.structure)
        self.velocity = 0.5
        self.p_best = list((np.inf, np.copy(self.weights)))

    def update_weights(self):
        self.layers = form_network(self.n_layer, np.copy(self.weights), self.structure)
        