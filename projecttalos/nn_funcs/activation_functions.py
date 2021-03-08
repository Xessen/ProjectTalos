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

def relu_backward(dA,cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
        
    return dZ
def tanh_backward(dA,cache):
    Z=cache
    t=(np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
    dZ=1-t**2

    return dZ

