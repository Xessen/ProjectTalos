import numpy as np
from .activation_functions import *

def compute_cost(AL,Y):
    m = Y.shape[1]
    cost = (-1/m)*np.sum(np.multiply(Y, np.log(AL))+np.multiply(1-Y, np.log(1-AL)))
    cost = np.squeeze(cost)   
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m)*np.dot(dZ,A_prev.T)
    db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
     
    return dA_prev, dW, db

def backward_activation(dA,cache,activation):
    linear_cache,activation_cache=cache

    dZ=eval(f"{activation}_backward(dA,activation_cache)")
    dA_prev, dW, db = linear_backward(dZ,linear_cache)

    return dA_prev, dW, db

def layer_seq(layers):
    sequence=[]
    for i in layers:
        b,a=i.split("*")[-1],i.split("*")[0]
        for y in range(int(b)):
            sequence.append(a)

    return sequence


def backward_model(AL,Y,caches,layer_func):
    grads = {}
    sequence=layer_seq(layer_func)
    print(sequence)
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) 
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = backward_activation(dAL, current_cache, activation = sequence[-1])
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = backward_activation(grads["dA" + str(l + 1)], current_cache, activation = sequence[l])
        
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters
