from forward_prop import *


class NeuralNetwork:
    def __init__(self,layer_dims,layer_func,learning_rate,iteration):
        self.layer_func=layer_func
        self.layer_dims=layer_dims
        self.learning_rate=learning_rate
        self.iteration=iteration
    def train(self,X,Y):
        costs=[]
        parameters=param_init(self.layer_dims)
        
        for i in range(0,self.iterations):

            AL, caches = forward_model(X,parameters,self.layer_func)

            cost = compute_cost(AL,Y)

            grads = backward_model(AL,Y,caches,self.layer_func)

            parameters = update_parameters(parameters,grads,self.learning_rate)

            if i%100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
                costs.append(cost)

