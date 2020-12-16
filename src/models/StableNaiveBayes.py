from abc import ABC, abstractmethod
from math import log
from scipy.sparse import csr_matrix
import numpy as np

'''
abstract naive bayes class, but numerically stable (uses logs of probabilities)
we assume y integer in {0, 1} (binary classification)
'''
class StableNaiveBayes(ABC):
    
    @abstractmethod
    def train(self, X, y):
        ''' X = sparse matrix of training vectors ; y = numpy array of classes'''
        pass

    @abstractmethod
    def p_xi_given_y(self, xi, i, y):
        ''' compute probability of X[i]=xi given y '''
        pass
    
    @abstractmethod
    def p_y(self, y):
        ''' compute probability of y '''
        pass
    
    @abstractmethod
    def reset_params(self):
        ''' reset parameters (needed for kfold) '''
        pass

    def log_p_xi_given_y(self, xi, i, y):
        ''' just log of p_xi_given_y (some implementations might want to override it, for example
            to pass from exponentiation inside p_xi_given_y to multiplication here)'''
        return np.log(self.p_xi_given_y(xi, i, y))
    
    def log_prob_y_given_x(self, X, y):
        ''' computes (dropping the denominator) log(P(y|X)) = 
        (bayes + den. drop) => log(P(X|y) P(y)) = sum(log(P(Xi|y)) + log(P(y)). Note: X is a single test, as numpy array  '''
        return sum([self.log_p_xi_given_y(X[i], i, y) for i in range(len(X))]) + log(self.p_y(y))
    
    def predict_class(self, X):
        ''' predict y for data X (single numpy array) '''
        p0 = self.log_prob_y_given_x(X, 0)
        p1 = self.log_prob_y_given_x(X, 1)
        return 0 if p0 > p1 else 1

    def multi_log_prob_y_given_x(self, X, y):
        ''' computes log P(y|Xi) for each row i in X. X can be a numpy array or a sparse matrix, the output must be a numpy array'''
        return np.array([self.log_prob_y_given_x(x, y) for x in X]) #slow in general, better provide other implementations

    def multi_predict_class(self, X):
        ''' predict classes (as a numpy array) for a matrix test data X (each row of the matrix is a test sample) [it might be a sparse matrix, or a numpy array if is small enough]'''
        return (self.multi_log_prob_y_given_x(X, 0) < self.multi_log_prob_y_given_x(X, 1)).astype('int')
        
    def multi_predict_class_from_score(self, scores, threshold=0.5):
        ''' given the scores computed with multi_prediction_score it predicts the labels (threshold=0.5 coincides with multi_predict_class)'''
        return (scores >= threshold).astype('int')

    def multi_prediction_score(self, X):
        ''' returns numpy array with score of class being 1 for each row of X (can be used for roc-curve) '''
        p0 = self.multi_log_prob_y_given_x(X, 0)
        p1 = self.multi_log_prob_y_given_x(X, 1)
        #p0 and p1 are always negative, doing p1/(p0+p1) gives higher score to the less probable class
        return 1 - p1 / (p0 + p1)