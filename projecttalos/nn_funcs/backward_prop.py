import numpy as np

def compute_cost(AL,Y):
    m = Y.shape[1]
    cost = (-1/m)*np.sum(np.multiply(Y, np.log(AL))+np.multiply(1-Y, np.log(1-AL)))
    cost = np.squeeze(cost)   
    return cost