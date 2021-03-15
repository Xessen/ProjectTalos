from projecttalos.nn_funcs.forward_prop import param_init,forward_model
from projecttalos.nn_funcs.backward_prop import backward_model,update_parameters,compute_cost
from projecttalos.data_preprocessing import ImagePreprocess
import numpy as np



class NeuralNetwork:
    """
    :This class takes 4 parameters:

    :param layer_dims: Initializing the structure of neural network with a list [input_layer,layer_1,...,layer_n]
    :param layer_func: Initializing the activation functions for each layer with a list ["sigmoid*3","relu*2","tanh*2","sigmoid*1"]
    :param learning_rate: Learning rate of model
    :param iteration: Number of iterations
 
    """
    def __init__(self,layer_dims,layer_func,learning_rate,iteration,train_score=None,test_score=None):

        self.layer_func=layer_func
        self.layer_dims=layer_dims
        self.learning_rate=learning_rate
        self.iteration=iteration
        self.test_score=test_score
        self.train_score=train_score
    def train(self,X,Y):
        """
        :Trains the model by given X and Y values:
        :This function takes 2 parameters:

        :param X: Input features of numpy arrays X.shape should be (number_of_features,number_of_training_values)
        :param Y: Output values of numpy arrays Y.shape should be (output_value,number_of_training_values) 
        
        :return: Returns W and b values which can be used in predict and score function
        
        """
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
        """
        :Predicts the output by given X and parameters:
        :This function takes 2 parameters:

        :param X: Input features of numpy arrays X.shape should be (number_of_features,number_of_training_values)
        :param parameters: Dictionary that holds W and b values

        :return: Returns output value


        """
        result,cache=forward_model(X,parameters,self.layer_func)
        return result
    def score(self,X,Y,parameters):
        """
        
        :This function takes 2 parameters:

        :param X: Input features of numpy arrays X.shape should be (number_of_features,number_of_training_values)
        :param Y: Output values of numpy arrays Y.shape should be (output_value,number_of_training_values)
        :param parameters: Dictionary that holds W and b values

        :return: Returns score of the model by given parameters


        """
        result,cache=forward_model(X,parameters,self.layer_func)
        self.test_score=np.divide(np.sum(Y)-np.mean(np.abs(np.subtract(Y,result))),np.sum(Y))*100
        print(f"Accuracy:{self.test_score}")


class LinearRegression:
    def __init__(self,coefs=None,score=None,intercept=None):
        self.coefs=coefs
        self.score=score
        self.intercept=intercept

    def train(self,X,Y):
        weights=np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y))
        self.intercept=weights[0]
        self.coefs=weights[1:]
        print(weights)

