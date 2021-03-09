from nn_funcs.forward_prop import *
from nn_funcs.backward_prop import *
from data_preprocessing import ImagePreprocess
x_train,x_test,y_train,y_test=ImagePreprocess(["test0","test1"])

class NeuralNetwork:
    def __init__(self,layer_dims,layer_func,learning_rate,iteration):
        self.layer_func=layer_func
        self.layer_dims=layer_dims
        self.learning_rate=learning_rate
        self.iteration=iteration
    def train(self,X,Y):
        costs=[]
        parameters=param_init(self.layer_dims)
        for i in range(0,self.iteration):
            AL, caches = forward_model(X,parameters,self.layer_func)
            cost = compute_cost(AL,Y)
            grads = backward_model(AL,Y,caches,self.layer_func)

            parameters = update_parameters(parameters,grads,self.learning_rate)

            if i%100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
                costs.append(cost)
        return parameters

nn=NeuralNetwork([19200,3,2,1],["relu*2","sigmoid*1"],0.005,300)
param=nn.train(x_train,y_train)

