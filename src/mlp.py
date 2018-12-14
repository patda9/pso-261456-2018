from activation_units import sigmoid

def calculate_loss(d, y_hats):
    return sum(abs(y_hats - d)) / d.shape[0]

def form_network(n_layer, weights, structure):
    layers = []
    i = 0
    for j in range(0, len(n_layer)):
        weights_shape = (int(n_layer[j] / int(structure[j])), int(structure[j]))
        layers.append(weights[i:n_layer[j]].reshape(weights_shape))
    i = n_layer[len(n_layer) - 1] + 1
    return layers

def forward_pass(d, individual, x, layer_pass=False):
    outputs = []
    if(layer_pass):
        layers = individual
    else:
        layers = individual.layers

    z = x.dot(layers[0])
    a = sigmoid(z)
    outputs.append(a)
    for w in layers[1:]:
        layer_input = z
        z = layer_input.dot(w)
        a = sigmoid(z)
        outputs.append(a)
    y_hats = outputs[-1]
    return y_hats
