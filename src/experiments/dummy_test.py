import numpy as np
from src.models.BernoulliNaiveBayes import BernoulliNaiveBayes
import math
from src.datasets.TwitterDSReader import TwitterDSReader
from scipy.sparse import csr_matrix
from sklearn.preprocessing import binarize

'''
print("Preprocessing test --- START\n")

ds = TwitterDSReader()
ds.read_from_file()
for i, tweet in enumerate(ds.docs()):
    for token in tweet.tokens:
        print(token, token.lemma_)
    if i==0: break

print("\nPreprocessing test --- END\n")
'''

y = np.array([1, 1, 0, 1, 0])
print(y, y.shape)
X = [[1, 0, 14, 1],
    [0, 0, 15, 1],
    [12, 0, 0, 0],
    [0, 0, 1, 0],
    [1, 14, 1, 1]]
X = np.array(X)
X = csr_matrix(X)
print(X)
print(X.shape)
binarize(X, copy=False)
print(X)


model = BernoulliNaiveBayes()
model.train(X, y)
print(model.m)
print(model.count_y_1)
print(model.count_x_1_y_1)
print(model.count_x_1_y_0)
'''
print(model.p_y(1), model.p_y(0))
print(model.p_xi_given_y([0, 0, 0, 0], 0, 0), model.p_xi_given_y([0, 0, 0, 0], 0, 1))

print(math.exp(model.log_prob_y_given_x([0, 0, 0, 0], 0)), math.exp(model.log_prob_y_given_x([0, 0, 0, 0], 1)))
print(model.predict_class([0, 0, 0, 0]))

print(math.exp(model.log_prob_y_given_x([1, 0, 0, 1], 0)), math.exp(model.log_prob_y_given_x([1, 0, 0, 1], 1)))
print(model.predict_class([1, 0, 0, 1]))
'''