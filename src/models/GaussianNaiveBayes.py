from src.models.StableNaiveBayes import StableNaiveBayes
import numpy as np
import math

class GaussianNaiveBayes(StableNaiveBayes):

    def __init__(self):
        pass

    def p_y_01(self, y):
        # Compute Prior Probabilities
        self.prior_1 = y.count(1) / len(y)
        self.prior_0 = y.count(0) / len(y)
        return self.p_y_1, self.p_y_0

    def train(self, embeddings, y):
        # Input: Embeddings (Tweet level), y array
        # Output: 
        # 1. Mean vector of all values in "positive" tweet's embeddings
        # 2. Mean vector of all values in "negative" tweet's embeddings
        # 3. Variance vector of all values in "positive" tweet's embeddings
        # 4. Variance vector of all values in "negative" tweet's embeddings
        self.y_1_indexes = [i[0] for i in enumerate(y) if y[i[0]] == 1]
        self.x_1_samples = [vector for vector in embeddings if embeddings.index(vector) in self.y_1_indexes]
        self.x_0_samples = [vector for vector in embeddings if embeddings.index(vector) not in self.y_1_indexes]
        self.x_1_mean = np.mean(self.x_1_samples, axis = 0)
        self.x_0_mean = np.mean(self.x_1_samples, axis = 0) 
        self.x_1_var = np.var(self.x_1_samples, axis = 0)
        self.x_0_var = np.var(self.x_0_samples, axis = 0)
        
        return self.x_1_mean , self.x_0_mean, self.x_1_var, self.x_0_var

    def compute_log_L(self, new_sample):
        #Input: New test sample
        #Output: y prediction {0, 1}

        data_x_1 = np.column_stack(new_sample, self.x_1_mean, self.x_1_var).transpose()
        likelihood_1 = np.apply_along_axis(compute_likelihood, data_x_1) #compute likelihood that P(xi = sample[i] | y=1) 
        log_l_1 = [np.log(like) for like in likelihood_1]   #compute log likelihoods
        p_y_1 = np.sum(log_l_1) + np.log(self.prior_1) #sum up log likelihoods and add prior

        data_x_0 = np.column_stack(new_sample, self.x_0_mean, self.x_0_var).transpose()
        likelihood_1 = np.apply_along_axis(compute_likelihood, data_x_0) #compute likelihood that P(xi = sample[i] | y=0)
        log_l_0 = [np.log(like) for like in likelihood_0] #compute log likelihoods
        p_y_0 = np.sum(log_l_0) + np.log(self.prior_0) #sum up log likelihoods and add prior

        if p_y_1 >= p_y_0:
            return 1
        else
            return 0

    def compute_likelihood(self, data):
        return 1/((2*math.pi*data[2])**0.5)*np.exp(-(data[0] - data[1])**2/(2*data[2]))





    
    
        
