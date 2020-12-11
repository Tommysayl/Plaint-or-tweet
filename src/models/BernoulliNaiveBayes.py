from src.models.StableNaiveBayes import StableNaiveBayes
import numpy as np
from scipy.sparse import csr_matrix

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

    #X must be a sparse matrix (csr_matrix) and y must be a numpy array
    #each row of X is a training data
    #Note: you can also call this many times (separating the dataset into batches)
    def train(self, X, y):
        if self.count_x_1_y_0 is None: #first time called
            self.count_x_1_y_0 = np.zeros(X.shape[1])
            self.count_x_1_y_1 = np.zeros(X.shape[1])

        self.m += len(y)
        self.count_y_1 += np.count_nonzero(y) #nonzero == 1

        y_col = np.transpose([y]) #y as a column vector
        sm_rows = X.multiply(csr_matrix(y_col)).sum(axis=0) #multiply each column of X by y (ie: consider only class 1 samples), and then sum all the rows
        self.count_x_1_y_1 += sm_rows.A1 #sm_rows is numpy matrix, A1 takes only the row
        sm_rows = (X.multiply(csr_matrix(1 - y_col))).sum(axis=0) #multiply each column of X by (1 - y) (ie: consider only class 0 samples), and then sum all the rows
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