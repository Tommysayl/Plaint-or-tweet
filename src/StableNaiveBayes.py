from abc import ABC, abstractmethod
from math import log
import numpy as np

'''
abstract naive bayes class, but numerically stable (uses logs of probabilities)
we assume y integer in {0, 1} (binary classification)
'''
class StableNaiveBayes(ABC):
    #X = list of training vectors
    #y = vector of classes
    @abstractmethod
    def train(self, X, y):
        pass

    #compute probability of X[i] given y
    @abstractmethod
    def p_xi_given_y(self, X, i, y):
        pass
    
    #compute probability of y
    @abstractmethod
    def p_y(self, y):
        pass
    
    '''
    computes (dropping the denominator) log(P(y|X)) = 
    (bayes + den. drop) => log(P(X|y) P(y)) = sum(log(P(Xi|y)) + log(P(y))
    '''
    def log_prob_y_given_x(self, X, y):
        return sum([log(self.p_xi_given_y(X, i, y)) for i in range(len(X))]) + log(self.p_y(y))
    
    #predict y for data X
    def predict_class(self, X):
        p0 = self.log_prob_y_given_x(X, 0)
        p1 = self.log_prob_y_given_x(X, 1)
        return 0 if p0 > p1 else 1
