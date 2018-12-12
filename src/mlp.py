from activation_units import sigmoid

def form_network(form, population):
    structure = form.split(',')
    weights = []
    
    i = 0
    n = 0
    for j in range(len(structure) - 1):
        structure[j] = int(structure[j])
        structure[j+1] = int(structure[j+1])
        n += structure[j] * structure[j+1]
        weights.append(population.solution[i:n].reshape(structure[j], structure[j+1]))
        i = n
    return weights

def forward_pass(d, form, population, x, model=False):
    y_hats = []
    if(model == True):
        optimal_solution = population
        outputs = []
        weights = form_network(form, optimal_solution)

        z = x.dot(weights[0])
        a = sigmoid(z)
        outputs.append(a)
        for w in weights[1:]:
            input = z
            z = input.dot(w)
            a = sigmoid(z)
            outputs.append(a)
        y_hats.append(outputs[-1])
    else:
        for i in range(len(population)):
            weights = form_network(form, population[i])
            outputs = []

            z = x.dot(weights[0])
            a = sigmoid(z)
            outputs.append(a)
            for w in weights[1:]:
                input = z
                z = input.dot(w)
                a = sigmoid(z)
                outputs.append(a)
            y_hats.append(outputs[-1])
    return y_hats
    