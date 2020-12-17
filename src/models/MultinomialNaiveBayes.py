from src.models.BernoulliNaiveBayes import BernoulliNaiveBayes
from math import log
import numpy as np

"""
Naive Bayes where:
each feature Xi has domain in {0,1,2,...}
and each y is in {0,1}

Note: Bernoulli performs the correct sums (in count_x_1_y_?) if we pass to it non-binarized data (in fact, counting of 1's is performed as a sum)
"""
class MultinomialNaiveBayes(BernoulliNaiveBayes):

    def __init__(self):
        super().__init__()
        self.count_y_1_words = 0 #number of words that appears in y=1 training cases 
        self.count_y_0_words = 0 #similar, for y=0

        self.d = 0

    def train(self, X, y):
        super().train(X, y)

        self.d = X.shape[1] #number of features
        self.count_y_1_words += self.count_x_1_y_1.sum() 
        self.count_y_0_words += self.count_x_1_y_0.sum()
        
        '''
        print(max(self.count_x_1_y_0))
        print(max(self.count_x_1_y_1))
        print(min(self.count_x_1_y_0))
        print(min(self.count_x_1_y_1))
        print(self.count_y_0_words)
        print(self.count_y_0_words)
        '''

    def p_xi_given_y(self, xi, i, y):
        #note that the return value should be raised to xi
        #but we handle this in log_p_xi_given_y as this exponentiation is not numerically stable
        if y == 1:
            return (self.count_x_1_y_1[i] + 1) / (self.count_y_1_words + self.d) #+1/+d for laplace smoothing
        return (self.count_x_1_y_0[i] + 1) / (self.count_y_0_words + self.d)
    
    def log_p_xi_given_y(self, xi, i, y):
        return xi * super().log_p_xi_given_y(min(xi, 1), i, y)
    
    def multi_log_prob_y_given_x(self, X, y):
        '''X must be a sparse csr matrix (each row is a test sample), otherwise it's unfeasable. this will output a numpy array with the log prob of y for each sample'''
        lpx1_y = np.array([ self.log_p_xi_given_y(1, i, y) for i in range(X.shape[1]) ]) #log P(x_i=1|y) for each i
        log_prob = np.full(shape=X.shape[0], fill_value=log(self.p_y(y))) #we start assuming all test data is made by zeroes
        
        #performs: Xri * log(P(x_i=1|y)) for each row r, and column i (and sums elements on rows)
        #and note Xri * log(P(x_i=1|y)) = log(P(x_i=1|y)^Xri) = log(P(x_i=Xri|y))
        if isinstance(X, np.ndarray):
            correction = np.dot(X, lpx1_y)
        else:
            correction = X.dot(lpx1_y) 
        
        return log_prob + correction #we get P(X|y)P(y)

    def reset_params(self):
        super().reset_params()
        self.count_y_1_words = 0 
        self.count_y_0_words = 0 
        self.d = 0