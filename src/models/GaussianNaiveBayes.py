from src.models.StableNaiveBayes import StableNaiveBayes
import numpy as np
import math

class GaussianNaiveBayes(StableNaiveBayes):

    def __init__(self):
        self.count_y_1 = 0
        self.m = 0

    def reset_params(self):
        pass

    def p_xi_given_y(self, xi, i, y):
        if y == 1:
            data = np.array((self.x_1_mean[i], self.x_1_var[i], xi))
        else:
            data = np.array((self.x_0_mean[i], self.x_0_var[i], xi ))
        likelihood = np.apply_along_axis(self.compute_likelihood, 0, data) 
        return likelihood

    def p_y(self, y):
        p1 = (self.count_y_1 + 1) / (self.m + 2) #+1/+2 are due to Laplace smoothing
        return p1 if y == 1 else (1 - p1)

    def train(self, embeddings, y):
        # Input: Embeddings (Tweet level), y array
        # Output: saves to self:
        # 1. Mean vector of all values in "positive" tweet's embeddings
        # 2. Mean vector of all values in "negative" tweet's embeddings
        # 3. Variance vector of all values in "positive" tweet's embeddings
        # 4. Variance vector of all values in "negative" tweet's embeddings
        self.m = len(y)
        self.count_y_1 += np.count_nonzero(y) #nonzero == 1
        self.y_1_indexes = [i[0] for i in enumerate(y) if y[i[0]] == 1]
        self.y_0_indexes = [i[0] for i in enumerate(y) if y[i[0]] == 0]
        self.x_1_samples = [embeddings[ind] for ind in self.y_1_indexes]
        self.x_0_samples = [embeddings[ind] for ind in self.y_0_indexes]
        self.x_1_mean = np.mean(self.x_1_samples, axis = 0)
        self.x_0_mean = np.mean(self.x_0_samples, axis = 0) 
        self.x_1_var = np.var(self.x_1_samples, axis = 0)
        self.x_0_var = np.var(self.x_0_samples, axis = 0)


    def multi_log_prob_y_given_x(self, X, y):
        #note that for a single sample Xi, we have:
        #log P(Xij|y) = log(1/sqrt(2*pi*var)) - (Xij - mean)^2 / 2*var = -log(sqrt(2*pi*var)) - (Xij - mean)^2 / 2*var
        log_prob = np.full(X.shape[0], np.log(self.p_y(y))) #start adding log P(y)
        mean = self.x_1_mean if y == 1 else self.x_0_mean #mean and variance to use, according to y
        var = self.x_1_var if y == 1 else self.x_0_var

        log_prob -= np.log(np.sqrt(2 * np.pi * var)).sum() #this is always added, indipendently of X
        XminusMean = np.apply_along_axis(lambda x: (x-mean)*(x-mean) / (2 * var), 1, X) #apply on each row

        #all features are summed (by naive bayes assumption), ie: sum each row and then add the resulting column vector to log_prob of each sample
        log_prob -= XminusMean.sum(axis=1) 
        return log_prob

    def multi_prediction_score(self, X):
        ''' returns numpy array with score of class being 1 for each row of X (can be used for roc-curve) '''
        p0 = self.multi_log_prob_y_given_x(X, 0)
        p1 = self.multi_log_prob_y_given_x(X, 1)
        M = max(max(p0), max(p1))
        if M > 0: #log prob can be positive in this case, but we assume it to be log of probability so it must be negative
            p0 -= M
            p1 -= M
        #p0 and p1 are always negative, doing p1/(p0+p1) gives higher score to the less probable class
        return 1 - p1 / (p0 + p1)
    
    def compute_likelihood(self, data):
        return 1/((2*math.pi*data[1])**0.5)*np.exp(-(data[2] - data[0])**2/(2*data[1]))