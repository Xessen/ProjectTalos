from nn_funcs.forward_prop import *
from nn_funcs.backward_prop import *
from data_preprocessing import ImagePreprocess
x_train,x_test,y_train,y_test=ImagePreprocess(["test0","test1"])
print(y_test)

class NeuralNetwork:
    def __init__(self,layer_dims,layer_func,learning_rate,iteration,train_score=None,test_score=None):
        self.layer_func=layer_func
        self.layer_dims=layer_dims
        self.learning_rate=learning_rate
        self.iteration=iteration
        self.test_score=test_score
        self.train_score=train_score
    def train(self,X,Y):
        costs=[]
        parameters=param_init(self.layer_dims)
        for i in range(self.iteration+1):
            AL, caches = forward_model(X,parameters,self.layer_func)
            cost = compute_cost(AL,Y)
            grads = backward_model(AL,Y,caches,self.layer_func)

            parameters = update_parameters(parameters,grads,self.learning_rate)
            costs.append(cost)
            if i%100 == 0:
                print (f"Cost after iteration {i}: {cost}")
        self.train_score=np.divide((np.sum(costs)-np.mean(costs)),np.sum(costs))*100
        print(f"Training Accuracy: {self.train_score}")    
        return parameters
    def predict(self,X,parameters):
        result,cache=forward_model(X,parameters,self.layer_func)
        return result
    def score(self,X,Y,parameters):
        result,cache=forward_model(X,parameters,self.layer_func)
        self.test_score=np.divide(np.sum(Y)-np.mean(np.abs(np.subtract(Y,result))),np.sum(Y))*100
        print(f"Accuracy:{self.test_score}")
        


nn=NeuralNetwork([19200,16,4,8,1],["relu*3","sigmoid*1"],0.005,1000)
param=nn.train(x_train,y_train)

nn.score(x_test,y_test,param)
