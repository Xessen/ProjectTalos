import numpy as np

def sigmoid(x):
    z = (1/(1 + np.exp(-x)))
    return z

def binary_step(x):
    if x<0:
        return 0
    else:
        return 1

def tanh(x):
    z = (2/(1 + np.exp(-2*x))) -1
    return z

def relu(x):
    if x<0:
        return 0
    else:
        return x

def elu(x, a):
    if x<0:
        return a*(np.exp(x)-1)
    else:
        return x

def softmax(x):
    z = np.exp(x)
    z_ = z/z.sum()
    return z_