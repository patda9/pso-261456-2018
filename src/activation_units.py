from numpy import exp

def sigmoid(x, dx=False):
    if(dx == True):
        return x * (1 - x)
    return 1 / (1 + exp(-x))
    