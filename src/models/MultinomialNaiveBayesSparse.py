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

    def p_xi_given_y(self, xi, i, y):
        #note that the return value should be raised to xi
        #but we handle this in log_p_xi_given_y as this exponentiation is not numerically stable

        if y == 1:
            return (self.count_x_1_y_1[i] + 1) / (self.count_y_1_words + self.d) #+1/+d for laplace smoothing
        return (self.count_x_1_y_0[i] + 1) / (self.count_y_0_words + self.d)
    
    def log_p_xi_given_y(self, xi, i, y):
        return xi * super().log_p_xi_given_y(xi, i, y)
    
    def multi_predict_class(self, X):
        # note that this works: 
        # even if we consider only the case for xi=0 (which will always give 0 in log-prob) and xi=1
        # when we do the dot product with X, we multiply each result for xi=1 by Xri which will result in adding log_p_xi_given_y for the correct Xri
        #return super().multi_predict_class(X)

        #calling super is equivalent to this: (but slightly less perfoming)
        lpx1_y0 = np.array([ self.log_p_xi_given_y(1, i, 0) for i in range(X.shape[1]) ]) #P(x_i=1|y=0) for each i
        lpx1_y1 = np.array([ self.log_p_xi_given_y(1, i, 1) for i in range(X.shape[1]) ]) #P(x_i=1|y=1) for each i
        
        log_prob_0 = np.full(shape=X.shape[0], fill_value=log(self.p_y(0))) #we start assuming all test data is made by zeroes
        log_prob_1 = np.full(shape=X.shape[0], fill_value=log(self.p_y(1)))

        #performs: Xri * log(P(x_i=1|y=0)) for each row(/test) r, and column(/feature) i (and sums elements on rows)
        #and note Xri * log(P(x_i=1|y=0)) = log(P(x_i=1|y=0)^Xri) = log(P(x_i=Xri|y=0))
        log_prob_0 = log_prob_0 + X.dot(lpx1_y0) 
        log_prob_1 = log_prob_1 + X.dot(lpx1_y1) #similar, for y=1

        return (log_prob_0 < log_prob_1).astype('int')