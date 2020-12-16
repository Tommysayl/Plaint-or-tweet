from src.models.StableNaiveBayes import StableNaiveBayes
from scipy.sparse import csr_matrix
from math import log
import numpy as np

'''
Naive Bayes where:
each feature Xi has domain in {0,1}
and each y is in {0,1}
'''
class BernoulliNaiveBayes(StableNaiveBayes):
    def __init__(self):
        self.m = 0
        self.count_y_1 = 0 #numer of training examples where y=1
        self.count_x_1_y_1 = None #i-th element is number of training examples where (Xi=1 and y=1) (will be numpy array)
        self.count_x_1_y_0 = None #i-th element is number of training examples where (Xi=1 and y=0)

    def train(self, X, y):
        '''
        X must be a sparse matrix (csr_matrix) and y must be a numpy array
        each row of X is a training data
        Note: you can also call this many times (separating the dataset into batches)
        '''
        if self.count_x_1_y_0 is None: #first time called
            self.count_x_1_y_0 = np.zeros(X.shape[1])
            self.count_x_1_y_1 = np.zeros(X.shape[1])

        self.m += len(y)
        self.count_y_1 += np.count_nonzero(y) #nonzero == 1

        y_col = np.transpose([y]) #y as a column vector
        sm_rows = X.multiply(csr_matrix(y_col)).sum(axis=0) #multiply each column of X by y (ie: consider only class 1 samples), and then sum all the rows
        self.count_x_1_y_1 += sm_rows.A1 #sm_rows is numpy matrix, A1 takes only the row
        sm_rows = X.multiply(csr_matrix(1 - y_col)).sum(axis=0) #multiply each column of X by (1 - y) (ie: consider only class 0 samples), and then sum all the rows
        self.count_x_1_y_0 += sm_rows.A1

    def p_xi_given_y(self, xi, i, y):
        p1 = 0
        if y == 1:
            p1 = (self.count_x_1_y_1[i] + 1) / (self.count_y_1 + 2) #+1/+2 are due to Laplace smoothing
        else:
            p1 = (self.count_x_1_y_0[i] + 1) / (self.m - self.count_y_1 + 2) #+1/+2 are due to Laplace smoothing

        return p1 if xi == 1 else (1 - p1)

    def p_y(self, y):
        p1 = (self.count_y_1 + 1) / (self.m + 2) #+1/+2 are due to Laplace smoothing
        return p1 if y == 1 else (1 - p1)
    
    def multi_log_prob_y_given_x(self, X, y):
        '''X must be a sparse csr matrix (each row is a test sample), otherwise it's unfeasable. this will output a numpy array with the log prob of y for each sample'''
        lpx0_y = np.array([ self.log_p_xi_given_y(0, i, y) for i in range(X.shape[1]) ]) #log P(x_i=0|y) for each i
        lpx1_y = np.array([ self.log_p_xi_given_y(1, i, y) for i in range(X.shape[1]) ]) #log P(x_i=1|y) for each i
        lp0 = lpx0_y.sum() + log(self.p_y(y)) #log P(X=00..0|y)P(y)
        log_prob = np.full(shape=X.shape[0], fill_value=lp0) #we start assuming all test data is made by zeroes

        #now we need to correct the log probabilities: for each non zero entry (r, c) in X
        #we need to subtract log(P(Xr_c=0|y)) and add log(P(Xr_c=1|y)) (because we assumed this entry to be 0, but now we need to correct this assumption)
        correction = X.dot(lpx1_y - lpx0_y) #performs: Xri * (P(x_i=1|y) - P(x_i=0|y)) for each row(/test) r, and column(/feature) i (and sums elements on rows)
        
        return log_prob + correction #we get P(X|y)P(y)
