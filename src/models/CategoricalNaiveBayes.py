from src.models.StableNaiveBayes import StableNaiveBayes
import numpy as np

"""
Naive Bayes where:
each feature Xi has domain in {0,1,2,...}
and each y is in {0,1}
"""


class CategoricalNaiveBayes(StableNaiveBayes):
    # cat_nums is the list of numbers of possible categories for
    # every feature (it must be of the same length as the feature
    # vector; categories start at 0, so make sure to pass
    # (max necessary category) + 1).
    def __init__(self, cat_nums=None):
        self.cat_nums = cat_nums
        # Dataset dimension
        self.m = 0
        # Number of training examples where y=1
        self.count_y_1 = 0
        # Number of training examples where y=0
        self.count_y_0 = 0

        # 1D vector -> [p_y0, p_y1]
        self.th1 = []

        # 3D vector [yi][feature_index][categorical_value]
        # -> [[px00_given_y0, px10_given_y0, px20_given_y0, ... ],
        #      [px01_given_y0, px11_given_y0, px21_given_y0, ... ],
        #       ...],
        #   [px00_given_y1, px10_given_y1, px20_given_y1, ... ],
        #      [px01_given_y1, px11_given_y1, px21_given_y1, ... ],
        #       ...]
        # Starting with 2 empty vectors, one for y0 and one for y1
        self.th2 = [[], []]

    # X and y must be numpy arrays
    # each row of X is a training example
    # Trains the two model parameter types:
    # theta1[i] -> N(yi)/m
    # theta2[i][j][l] -> N(xlj, yi)/N(yi)
    def train(
        self,
        X,
        y,
    ):

        self.m = len(y)
        self.count_y_1 = np.count_nonzero(y)  # nonzero == 1
        self.count_y_0 = self.m - self.count_y_1

        # Training theta1
        self.th1.append((self.count_y_0 + 1) / (self.m + 2))
        self.th1.append((self.count_y_1 + 1) / (self.m + 2))

        # Numbers categorical values
        cat_nums = (
            self.cat_nums
            if self.cat_nums is not None
            else [np.max(X[:, i]) + 1 for i in range(X.shape[1])]
        )  # Also considering 0 as a value

        # Initializing theta2
        for i in range(X.shape[1]):
            self.th2[0].append(np.zeros(cat_nums[i]))
            self.th2[1].append(np.zeros(cat_nums[i]))

        # Training theta2
        for j in range(X.shape[1]):
            for k in range(cat_nums[j]):
                self.th2[0][j][k] = np.count_nonzero((X[:, j] == k) & (y == 0))
                self.th2[1][j][k] = np.count_nonzero((X[:, j] == k) & (y == 1))
        """
        for i in range(self.m):
            # print((i * 100) / self.m, "%")
            for j in range(X.shape[1]):
                # Obtaining N(xlj, yi)
                self.th2[y[i]][j][X[i][j]] += 1"""

        # Applying Laplace smoothing
        add = (1 / np.array(cat_nums)) * X.shape[1]
        div0 = self.count_y_0 + X.shape[1]
        div1 = self.count_y_1 + X.shape[1]
        for j in range(X.shape[1]):
            self.th2[0][j] = (self.th2[0][j] + add[j]) / div0
            self.th2[1][j] = (self.th2[1][j] + add[j]) / div1

    def p_xi_given_y(self, xi, i, y):
        return self.th2[y][i][xi]

    def p_y(self, y):
        return self.th1[y]
    
    def reset_params(self):
        self.m = 0
        self.count_y_1 = 0
        self.count_y_0 = 0
        self.th1 = []
        self.th2 = [[], []]

    def multi_predict_class(self, X):
        return np.array([self.predict_class(x) for x in X]) #slow!!!
