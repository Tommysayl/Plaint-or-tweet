from src.models.StableNaiveBayes import StableNaiveBayes
import numpy as np

"""
Naive Bayes where:
each feature Xi has domain in {0,1,2,...}
and each y is in {0,1}
"""


class CategoricalNaiveBayes(StableNaiveBayes):
    def __init__(self):
        # Dataset dimension
        self.m = 0
        # Number of training examples where y=1
        self.count_y_1 = 0
        # Number of training examples where y=0
        self.count_y_0 = 0

        # 1D vector -> [p_y0, p_y1]
        self.th1 = []

        # 3D vector [yi][categorical_value][feature_index]
        # -> [[px00_given_y0, px01_given_y0, px02_given_y0, ... ],
        #      [px10_given_y0, px11_given_y0, px12_given_y0, ... ],
        #       ...],
        #   [px00_given_y1, px01_given_y1, px02_given_y1, ... ],
        #      [px10_given_y1, px11_given_y1, px12_given_y1, ... ],
        #       ...]
        # Starting with 2 empty vectors, one for y0 and one for y1
        self.th2 = [[], []]

    # X and y must be numpy arrays
    # each row of X is a training example
    # Trains the two model parameter types:
    # theta1[i] -> N(yi)/m
    # theta2[i][l][j] -> N(xlj, yi)/N(yi)
    def train(self, X, y):

        self.m = len(y)
        self.count_y_1 = np.count_nonzero(y)  # nonzero == 1
        self.count_y_0 = self.m - self.count_y_1

        # Training theta1
        self.th1.append((self.count_y_0 + 1) / (self.m + 2))
        self.th1.append((self.count_y_1 + 1) / (self.m + 2))

        # Number categorical values
        cat_num = np.max(X) + 1  # Also considering 0 as a value

        # Initializing theta2
        for i in range(cat_num):
            self.th2[0].append([0 for i in range(X.shape[1])])
            self.th2[1].append([0 for i in range(X.shape[1])])

        # Training theta2
        for i in range(self.m):
            # print((i * 100) / self.m, "%")
            for j in range(X.shape[1]):
                # Obtaining N(xlj, yi)
                self.th2[y[i]][X[i][j]][j] += 1

        for i in range(cat_num):
            for j in range(X.shape[1]):
                self.th2[0][i][j] = (self.th2[0][i][j] + (1 / cat_num) * X.shape[1]) / (
                    self.count_y_0 + X.shape[1]
                )  # Applying Laplace smoothing
                self.th2[1][i][j] = (self.th2[1][i][j] + (1 / cat_num) * X.shape[1]) / (
                    self.count_y_1 + X.shape[1]
                )  # same here

    def p_xi_given_y(self, xi, i, y):
        return self.th2[y][xi][i]

    def p_y(self, y):
        return self.th1[y]
    
    def multi_predict_class(self, X):
        return np.array([self.predict_class(x) for x in X]) #slow!!!