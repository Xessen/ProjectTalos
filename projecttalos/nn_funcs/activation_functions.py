import numpy as np

def sigmoid(x):
    z = (1/(1 + np.exp(-x)))
    cache=x
    return z,cache

def binary_step(x):
    if x<0:
        return 0
    else:
        return 1

def tanh(x):
    z = (2/(1 + np.exp(-2*x))) -1
    cache=x
    return z,cache

def relu(x):
    z = np.maximum(0,x)
    cache=x
    return z,cache

def elu(x, a):
    if x<0:
        return a*(np.exp(x)-1)
    else:
        return x

def softmax(x):
    z = np.exp(x)
    z_ = z/z.sum()
    cache=x
    return z_,cache