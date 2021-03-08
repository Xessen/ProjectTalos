from activation_functions import *
import numpy as np


def param_init(layer_dims):
    parameters = {}
    L = len(layer_dims)          
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        
    return parameters


def forward_activation(A_,W,b,activation):
    Z=np.dot(W,A_)+b
    linear_cache=(A_,W,b)
    A,activation_cache=eval(f"{activation}(Z)")
    cache=(linear_cache,activation_cache)
    return A,cache

def forward_model(X,parameters,layer_func):
    caches = []
    counter=1
    A = X
    numberofcalc=len(layer_func)   
    for i in range(numberofcalc):
        layerfunc,layerrepeat=layer_func[i].split("*")[0],layer_func[i].split("*")[1]
        for y in range(int(layerrepeat)):
            if i==numberofcalc-1 and y==int(layerrepeat)-1:
                AL,cache=forward_activation(A,parameters["W"+str(counter)],parameters["b"+str(counter)],layerfunc)
                caches.append(cache)
                counter+=1
            else:
                A_prev=A
                A,cache=forward_activation(A_prev,parameters["W"+str(counter)],parameters["b"+str(counter)],layerfunc)
                caches.append(cache)
                counter+=1
    return AL,caches

