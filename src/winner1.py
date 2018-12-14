import numpy as np

def form_network(n_layer, weights, structure):
    layers = []
    i = 0
    for j in range(0, len(n_layer)):
        weights_shape = (int(n_layer[j] / int(structure[j])), int(structure[j]))
        layers.append(weights[i:n_layer[j]].reshape(weights_shape))
    print(j)
    i = n_layer[j] + 1
    return layers

weights = np.random.randn(58, 1)
n_layer = [40, 15, 3]
structure = [5, 3, 1]

print(form_network(n_layer, weights, structure))