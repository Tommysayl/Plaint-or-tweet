import numpy as np
from src.models.BernoulliNaiveBayes import BernoulliNaiveBayes
from src.models.MultinomialNaiveBayesSparse import MultinomialNaiveBayes
import math
from src.datasets.TwitterDSReader import TwitterDSReader
from scipy.sparse import csr_matrix
from sklearn.preprocessing import binarize
from src.utils import discretizeVector

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

print(model.p_y(1), model.p_y(0))
print(model.p_xi_given_y([0, 0, 0, 0], 0, 0), model.p_xi_given_y([0, 0, 0, 0], 0, 1))

print(model.log_prob_y_given_x([0, 0, 0, 0], 0), model.log_prob_y_given_x([0, 0, 0, 0], 1))
print(model.predict_class([0, 0, 0, 0]))

print(model.log_prob_y_given_x([1, 0, 0, 1], 0), model.log_prob_y_given_x([1, 0, 0, 1], 1))
print(model.predict_class([1, 0, 0, 1]))

print(model.log_prob_y_given_x([0, 1, 1, 0], 0), model.log_prob_y_given_x([0, 1, 1, 0], 1))
print(model.predict_class([0, 1, 1, 0]))

X_test = [[0, 0, 0, 0],
          [1, 0, 0, 1],
          [0, 1, 1, 0]]
X_test = csr_matrix(np.array(X_test))
print(model.sparse_predict_class(X_test))
'''

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


model = MultinomialNaiveBayes()
model.train(X, y)
print(model.m)
print(model.count_y_1)
print(model.count_x_1_y_1)
print(model.count_x_1_y_0)
print(model.count_y_0_words)
print(model.count_y_1_words)


print(model.log_prob_y_given_x([0, 0, 0, 0], 0), model.log_prob_y_given_x([0, 0, 0, 0], 1))
print(model.predict_class([0, 0, 0, 0]))

print(model.log_prob_y_given_x([1, 0, 0, 1], 0), model.log_prob_y_given_x([1, 0, 0, 1], 1))
print(model.predict_class([1, 0, 0, 1]))

print(model.log_prob_y_given_x([0, 1, 1, 0], 0), model.log_prob_y_given_x([0, 1, 1, 0], 1))
print(model.predict_class([0, 1, 1, 0]))

X_test = [[0, 0, 0, 0],
          [1, 0, 0, 1],
          [0, 1, 1, 0]]
X_test = csr_matrix(np.array(X_test))
print(model.sparse_predict_class(X_test))
'''

'''
#vt = np.array([23, 21, 5, 73, 6, 30])
#print(discretizeVector(vt, 6, 50, 5))
X_train_vec = [[23, 54, 96, 3, 53],
               [10, 20, 43, 42, 11],
               [30, 23, 54, 23, 4],
               [15, 34, 23, 15, 43]]

X_train_vec = np.array(X_train_vec)
print(X_train_vec)
minMaxPair = np.apply_along_axis(lambda x: (min(x), max(x)), 0, X_train_vec) #find (min, max) for each column of X_train_vec
print(minMaxPair)
X_train_vec = np.array([discretizeVector(v, minMaxPair[0][i], minMaxPair[1][i], 5) for i,v in enumerate(X_train_vec.T)]).T #discretize each column
print(X_train_vec)
'''

v = np.array([[1, 2, 3, 4],
             [5, 6, 7, 8],
             [12, 11, 10, 9]])

print(np.mean(v, axis=0))