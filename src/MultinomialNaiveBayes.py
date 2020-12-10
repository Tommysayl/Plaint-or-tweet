from StableNaiveBayes import StableNaiveBayes
import numpy as np

"""
Naive Bayes where:
each feature Xi has domain in {0,1,2,...}
and each y is in {0,1}
"""


class MultinomialNaiveBayes(StableNaiveBayes):
    def __init__(self):
        # Dataset dimension
        self.m = 0
        # Number of training examples where y=1
        self.count_y_1 = 0
        # Number of training examples where y=0
        self.count_y_0 = 0

        # 1D vector -> [p_y0, p_y1]
        self.th1 = []

        # 2D vector -> [[px0_given_y0, px1_given_y0, px2_given_y0, ... ],
        #              [px0_given_y1, px1_given_y1, px2_given_y1, ... ]]]
        self.th2 = []

    # X and y must be numpy arrays
    # each row of X is a training example
    # Trains the two model parameter types:
    # theta1[i] -> N(yi)/m
    # theta2[i][j] -> N(xj, yi)/N(yi)
    def train(self, X, y):

        self.m = len(y)
        self.count_y_1 = np.count_nonzero(y)  # nonzero == 1
        self.count_y_0 = self.m - self.count_y_1

        # Training theta1
        self.th1.append((self.count_y_0 + 1) / (self.m + 2))
        self.th1.append((self.count_y_1 + 1) / (self.m + 2))

        # Initializing theta2
        self.th2.append([0 for i in range(X.shape[1])])
        self.th2.append([0 for i in range(X.shape[1])])

        # Training theta2
        for i in range(self.m):
            for j in range(X.shape[1]):
                # Obtaining N(xj, yi)
                self.th2[y[i]][j] += X[i][j]

        for j in range(X.shape[1]):
            self.th2[0][j] = (self.th2[0][j] + 1) / (
                self.count_y_0 + X.shape[1]
            )  # Applying Laplace smoothing
            self.th2[1][j] = (self.th2[1][j] + 1) / (
                self.count_y_1 + X.shape[1]
            )  # same here

    def p_xi_given_y(self, X, i, y):
        return self.th2[y][i]

    def p_y(self, y):
        return self.th1[y]