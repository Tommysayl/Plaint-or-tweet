from StableNaiveBayes import StableNaiveBayes
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
        self.count_x_1_y_1 = None #i-th element is number of training examples where (Xi=1 and y=1) 
        self.count_x_1_y_0 = None #i-th element is number of training examples where (Xi=1 and y=0)

    #X and y must be numpy arrays
    #each row of X is a training data
    #Note: you can also call this many times (separating the dataset into batches)
    def train(self, X, y):
        if self.count_x_1_y_0 is None: #first time called
            self.count_x_1_y_0 = np.zeros(len(X[0]))
            self.count_x_1_y_1 = np.zeros(len(X[0]))

        self.m += len(y)
        self.count_y_1 += np.count_nonzero(y) #nonzero == 1
        self.count_x_1_y_1 += np.apply_along_axis(lambda x: np.count_nonzero(x * y), 0, X) #x*y => x=1 and y=1
        self.count_x_1_y_0 += np.apply_along_axis(lambda x: np.count_nonzero(x * (1-y)), 0, X) #x*(1-y) => x=1 and y=0

    def p_xi_given_y(self, X, i, y):
        p1 = 0
        if y == 1:
            p1 = (self.count_x_1_y_1[i] + 1) / (self.count_y_1 + 2) #+1/+2 are due to Laplace smoothing
        else:
            p1 = (self.count_x_1_y_0[i] + 1) / (self.m - self.count_y_1 + 2) #+1/+2 are due to Laplace smoothing

        return p1 if X[i] == 1 else (1 - p1)

    def p_y(self, y):
        p1 = (self.count_y_1 + 1) / (self.m + 2) #+1/+2 are due to Laplace smoothing
        return p1 if y == 1 else (1 - p1)