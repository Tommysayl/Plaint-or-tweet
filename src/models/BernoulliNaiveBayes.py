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
    
    def sparse_predict_class(self, X):
        ''' X must be a sparse csr matrix (each row is a test sample), this will output a numpy array with the predicted classes for each sample '''

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