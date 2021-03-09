from nn_funcs.forward_prop import *
from nn_funcs.backward_prop import *
from data_preprocessing import ImagePreprocess
x_train,x_test,y_train,y_test=ImagePreprocess(["test0","test1"])
print(y_test)

class NeuralNetwork:
    def __init__(self,layer_dims,layer_func,learning_rate,iteration,score=None):
        self.layer_func=layer_func
        self.layer_dims=layer_dims
        self.learning_rate=learning_rate
        self.iteration=iteration
        self.score=score
    def train(self,X,Y):
        costs=[]
        parameters=param_init(self.layer_dims)
        for i in range(0,self.iteration):
            AL, caches = forward_model(X,parameters,self.layer_func)
            cost = compute_cost(AL,Y)
            grads = backward_model(AL,Y,caches,self.layer_func)

            parameters = update_parameters(parameters,grads,self.learning_rate)

            if i%100 == 0:
                print (f"Cost after iteration {i}: {cost}")
                costs.append(cost)
        return parameters
    def predict(self,X,parameters):
        result,cache=forward_model(X,parameters,self.layer_func)
        return result

nn=NeuralNetwork([19200,16,8,1],["relu*2","sigmoid*1"],0.005,1000)
param=nn.train(x_train,y_train)

print(nn.predict(x_test,param))