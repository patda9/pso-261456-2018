import numpy as np

data_path = ['./src/dataset/aq-uci.csv', './src/dataset/aq-uci-norm.csv']

def k_fold(data, k=10):
    # separate
    d_shape = (data.shape[0], 1)
    x = np.hsplit(data, [data.shape[1] - 1])[0]
    d = np.hsplit(data, [data.shape[1] - 1])[1].reshape(d_shape)
    # folding
    fold_len = int(data.shape[0] / k)
    x_folds = []
    d_folds = []
    for i in range(k):
        if(i == k - 1): # len - 1 = idx
            x_folds += [x[i * fold_len:(i+1) * fold_len + (x.shape[0] % fold_len)]]
            d_folds += [d[i * fold_len:(i+1) * fold_len + (x.shape[0] % fold_len)]]
        else:
            x_folds += [x[i * fold_len:(i+1) * fold_len]]
            d_folds += [d[i * fold_len:(i+1) * fold_len]]

    return x_folds, d_folds

def get_data(data_path):
    return np.genfromtxt(data_path, delimiter=',')
