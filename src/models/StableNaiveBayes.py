from abc import ABC, abstractmethod
from math import log
import numpy as np

'''
abstract naive bayes class, but numerically stable (uses logs of probabilities)
we assume y integer in {0, 1} (binary classification)
'''
class StableNaiveBayes(ABC):
    
    @abstractmethod
    def train(self, X, y):
        ''' X = sparse matrix of training vectors ; y = numpy of classes'''
        pass

    @abstractmethod
    def p_xi_given_y(self, xi, i, y):
        ''' compute probability of X[i]=xi given y '''
        pass
    
    @abstractmethod
    def p_y(self, y):
        ''' compute probability of y '''
        pass
    
    def log_p_xi_given_y(self, xi, i, y):
        ''' just log of p_xi_given_y (some implementations might want to override it, for example
            to pass from exponentiation inside p_xi_given_y to multiplication here)'''
        return log(self.p_xi_given_y(xi, i, y))

    def sparse_predict_class(self, X):
        ''' X must be a sparse csr matrix (each row is a test sample), this will output a numpy array with the predicted classes for each sample '''
        lp0 = log(self.p_y(0)) + sum([self.log_p_xi_given_y(0, i, 0) for i in range(X.shape[1])]) #P(x=0|y=0)
        lp1 = log(self.p_y(1)) + sum([self.log_p_xi_given_y(0, i, 1) for i in range(X.shape[1])]) #P(x=0|y=1)
        log_prob_0 = np.full(shape=X.shape[0], fill_value=lp0) #we start assuming all test data is made by x=0
        log_prob_1 = np.full(shape=X.shape[0], fill_value=lp1)
        for row, col in zip(*X.nonzero()): #iterate over non-zero entries
            #we now know col-th feature of row-th test data is not 0, so we adjust log_prob_y[row]
            log_prob_0[row] -= self.log_p_xi_given_y(0, col, 0) #x_row[col] is 1, not 0 as we thought
            log_prob_0[row] += self.log_p_xi_given_y(1, col, 0)
            log_prob_1[row] -= self.log_p_xi_given_y(0, col, 1) #x_row[col] is 1, not 0 as we thought
            log_prob_1[row] += self.log_p_xi_given_y(1, col, 1)

        y_pred = (log_prob_0 < log_prob_1).astype('int') #log_prob_0 < log_prob_1 => class 1, else log_prob_0 >= log_prob_1 => class 0
        return y_pred
        
    def log_prob_y_given_x(self, X, y):
        ''' computes (dropping the denominator) log(P(y|X)) = 
        (bayes + den. drop) => log(P(X|y) P(y)) = sum(log(P(Xi|y)) + log(P(y)) '''
        return sum([self.log_p_xi_given_y(X[i], i, y) for i in range(len(X))]) + log(self.p_y(y))
    
    def predict_class(self, X):
        ''' predict y for data X '''
        p0 = self.log_prob_y_given_x(X, 0)
        p1 = self.log_prob_y_given_x(X, 1)
        return 0 if p0 > p1 else 1