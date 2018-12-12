from numpy import concatenate

def cross_validation(d_folds, x_folds, k=10):
    for i in range(k):
        # print('fold:', i)
        x_temp = x_folds.copy()
        d_temp = d_folds.copy()
        x_test = x_temp[i]
        d_test = d_temp[i]
        del(x_temp[i])
        del(d_temp[i])
        x_train = concatenate(x_temp, axis=0)
        d_train = concatenate(d_temp, axis=0)
    return [(x_train, d_train), (x_test, d_test)]
    