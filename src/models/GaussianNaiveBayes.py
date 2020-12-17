from src.models.StableNaiveBayes import StableNaiveBayes
import numpy as np
import math

class GaussianNaiveBayes(StableNaiveBayes):


    def __init__(self):
        self.y = 0

    def reset_params(self):
        pass


    def p_xi_given_y(self, xi, i, y):
        if y == 1:
            data = np.array((self.x_1_mean[i], self.x_1_var[i], xi))
            likelihood = np.apply_along_axis(self.compute_likelihood, 0, data) #compute likelihood that P(xi = sample[i] | y=1)
        else:
            data = np.array((self.x_0_mean[i], self.x_0_var[i], xi ))
            likelihood = np.apply_along_axis(self.compute_likelihood, 0, data) #compute likelihood that P(xi = sample[i] | y=0)
        return likelihood


    def p_y(self, y):

        py1 = np.bincount(self.y)
        return py1[1] / len(self.y) if y ==1 else py1[0] / len(self.y)



    def train(self, embeddings, y):
        # Input: Embeddings (Tweet level), y array
        # Output: saves to self:
        # 1. Mean vector of all values in "positive" tweet's embeddings
        # 2. Mean vector of all values in "negative" tweet's embeddings
        # 3. Variance vector of all values in "positive" tweet's embeddings
        # 4. Variance vector of all values in "negative" tweet's embeddings
        self.y = y
        self.y_1_indexes = [i[0] for i in enumerate(y) if y[i[0]] == 1]
        self.y_0_indexes = [i[0] for i in enumerate(y) if y[i[0]] == 0]
        self.x_1_samples = [embeddings[ind] for ind in self.y_1_indexes]
        self.x_0_samples = [embeddings[ind] for ind in self.y_0_indexes]
        self.x_1_mean = np.mean(self.x_1_samples, axis = 0)
        self.x_0_mean = np.mean(self.x_0_samples, axis = 0) 
        self.x_1_var = np.var(self.x_1_samples, axis = 0)
        self.x_0_var = np.var(self.x_0_samples, axis = 0)

        
        return None


    def compute_likelihood(self, data):
        return 1/((2*math.pi*data[2])**0.5)*np.exp(-(data[0] - data[1])**2/(2*data[2]))





    
    
        
