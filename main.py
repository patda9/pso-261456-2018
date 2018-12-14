import numpy as np
import sys
sys.path.insert(0, './src/')

import cross_validation as crv
import pso

from preprocess import data_path, get_data, k_fold

if(__name__ == '__main__'):
    dataset_index = [0, 1]
    # aq_uci_data5 = get_data(data_path[0])
    aq_uci_data5 = get_data(data_path[dataset_index[0]])
    aq_uci_data10 = get_data(data_path[dataset_index[1]])

    x5 = np.hsplit(aq_uci_data5, [aq_uci_data5.shape[1] - 1])[0]
    d5 = np.hsplit(aq_uci_data5, [aq_uci_data5.shape[1] - 1])[1]
    x10 = np.hsplit(aq_uci_data10, [aq_uci_data10.shape[1] - 1])[0]
    d10 = np.hsplit(aq_uci_data10, [aq_uci_data10.shape[1] - 1])[1]

    forms = ['3-1', '5-3-1']
    structure1, structure2 = forms[0].split('-'), forms[1].split('-')
    n_weights1, n_weights2 = 0, 0
    l_weights1, l_weights2 = [], []

    # n_weights1 += x10.shape[1] * int(structure1[0])
    # l_weights1.append(x10.shape[1] * int(structure1[0]))
    # for i in range(1, len(structure1)):
    #     n_weights1 += int(structure1[i-1]) * int(structure1[i])
    #     l_weights1.append(int(structure1[i-1]) * int(structure1[i]))
    # print(n_weights1, l_weights1)
    
    n_weights2 += x10.shape[1] * int(structure2[0])
    l_weights2.append(x10.shape[1] * int(structure2[0]))
    for i in range(1, len(structure2)):
        n_weights2 += int(structure2[i-1]) * int(structure2[i])
        l_weights2.append(int(structure2[i-1]) * int(structure2[i]))
    print(n_weights2, l_weights2)

    n = 50 # init number of particle number
    individuals = []
    for i in range(n):
        # individuals.append(pso.Individual(l_weights1, n_weights1, structure1))
        individuals.append(pso.Individual(l_weights2, n_weights2, structure2))
    # x5_folds, d5_folds = k_fold(aq_uci_data5)
    x10_folds, d10_folds = k_fold(aq_uci_data10)
    
    # crv.cross_validation(dataset_index[0], d5_folds, individuals, x5_folds)
    crv.cross_validation(dataset_index[1], d10_folds, individuals, x10_folds)
    
