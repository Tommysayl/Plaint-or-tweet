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
        #lp0 = log(self.p_y(0)) + sum([self.log_p_xi_given_y(0, i, 0) for i in range(X.shape[1])]) #P(x=0|y=0)
        #lp1 = log(self.p_y(1)) + sum([self.log_p_xi_given_y(0, i, 1) for i in range(X.shape[1])]) #P(x=0|y=1)
        
        lpx0_y0 = np.array([ self.log_p_xi_given_y(0, i, 0) for i in range(X.shape[1]) ]) #P(x_i=0|y=0) for each i
        lpx0_y1 = np.array([ self.log_p_xi_given_y(0, i, 1) for i in range(X.shape[1]) ]) #P(x_i=0|y=1) for each i
        lpx1_y0 = np.array([ self.log_p_xi_given_y(1, i, 0) for i in range(X.shape[1]) ]) #P(x_i=1|y=0) for each i
        lpx1_y1 = np.array([ self.log_p_xi_given_y(1, i, 1) for i in range(X.shape[1]) ]) #P(x_i=1|y=1) for each i

        lp0 = lpx0_y0.sum() + log(self.p_y(0)) #P(X=00..0|y=0)P(y=0)
        lp1 = lpx0_y1.sum() + log(self.p_y(1)) #P(X=00..0|y=1)P(y=0)
        log_prob_0 = np.full(shape=X.shape[0], fill_value=lp0) #we start assuming all test data is made by zeroes
        log_prob_1 = np.full(shape=X.shape[0], fill_value=lp1)

        #now we need to correct the log probabilities: for each non zero entry (r, c) in X
        #we need to subtract log(P(Xr_c=0|y)) and add log(P(Xr_c=1|y)) (because we assumed this entry to be 0, but now we need to correct this assumption)
        
        correction0 = X.dot(lpx1_y0 - lpx0_y0) #performs: Xri * (P(x_i=1|y=0) - P(x_i=0|y=0)) for each row(/test) r, and column(/feature) i (and sums elements on rows)
        correction1 = X.dot(lpx1_y1 - lpx0_y1) #similar, for y=1
        
        log_prob_0 = log_prob_0 + correction0 #we get P(X|y=0)P(y=0)
        log_prob_1 = log_prob_1 + correction1

        y_pred = (log_prob_0 < log_prob_1).astype('int') #if log_prob_0 < log_prob_1 => class 1; if log_prob_0 >= log_prob_1 => class 0
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