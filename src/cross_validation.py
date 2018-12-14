import numpy as np
import mlp

from preprocess import aq_min, aq_max

def cross_validation(data_set, d_folds, individuals, x_folds, k=10):
    if(data_set == 0):
        data_set = 5
    elif(data_set == 1):
        data_set = 10

    for i in range(k):
        file = open('./src/outputs/aq' + str(data_set) + 'fold-' + str(i) + ('-log.txt'), 'w')
        g_best = list((np.inf, individuals[0].weights))

        print('fold:', i)
        x_temp = x_folds.copy()
        d_temp = d_folds.copy()
        x_test = x_temp[i]
        d_test = d_temp[i]
        del(x_temp[i])
        del(d_temp[i])
        x_train = np.concatenate(x_temp, axis=0)
        d_train = np.concatenate(d_temp, axis=0)
        g_best = train(d_train, g_best, i, individuals, x_train)
        g_best_weights = mlp.form_network(individuals[0].n_layer, g_best[1], individuals[0].structure)
        mean_abs_e = test(d_test, g_best_weights, x_test)
        file.writelines('Best weights: ' + str(g_best_weights) + '\n')
        file.writelines('Best f(x): ' + str(g_best[0]) + '\n')
        file.writelines('MAE: ' + str(mean_abs_e) + '\n')
        file.close()

def train(d_train, g_best, i, individuals, x_train, epochs=64):
    r1, r2 = np.random.uniform(0, 1), np.random.uniform(0, 1)
    c1, c2 = np.random.uniform(0.1, 3), np.random.uniform(0.1, 1)
    p1 = r1 * c1
    p2 = r2 * c2

    for j in range(epochs):
        print('iteration:', j)
        for individual in individuals:
            y_hats = mlp.forward_pass(d_train, individual, x_train)
            individual.fx = mlp.calculate_loss(d_train, y_hats)
        
        for k in range(len(individuals)):
            print('individual:', k, individuals[k].fx, g_best[0])
            if(individuals[k].fx < individuals[k].p_best[0]):
                # print('update p_best:', individuals[k].fx - individuals[k].p_best[0])
                individuals[k].p_best[0] = np.copy(individual.fx)
            if(individuals[k].fx < g_best[0]):
                # print('update p_best:', individuals[k].fx - g_best[0])
                g_best[0] = individuals[k].fx
                print(g_best[0])

        for k in range(len(individuals)):
            individuals[k].velocity += (p1 * (individuals[k].p_best[1] - individuals[k].weights) + p2 * (g_best[1] - individuals[k].weights))
            individuals[k].weights += individuals[k].velocity
            individuals[k].update_weights()
        print(g_best[0])
    
    return g_best

def test(d_test, layers, x_test):
    y_hats = mlp.forward_pass(d_test, layers, x_test, layer_pass=True)
    mean_abs_e_test = mlp.calculate_loss(d_test, y_hats)
    return mean_abs_e_test