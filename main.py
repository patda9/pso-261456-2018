import numpy as np
import sys
sys.path.insert(0, './src/')

import cross_validation as crv
from preprocess import data_path, get_data, k_fold

if(__name__ == '__main__'):
    aq_uci_data = get_data(data_path[0])
    # print(aq_uci_data)
    aq_uci_data_norm = get_data(data_path[1])
    # print(aq_uci_data_norm)

    x_folds, d_folds = k_fold(aq_uci_data, 10)
    print(len(x_folds))
    print(len(d_folds))