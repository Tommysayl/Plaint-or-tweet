import numpy as np
from BernoulliNaiveBayes import BernoulliNaiveBayes
import math
from TwitterDSReader import TwitterDSReader

print("Preprocessing test --- START\n")

ds = TwitterDSReader()
ds.read_from_file()
for i, tweet in enumerate(ds.docs()):
    for sentence in tweet.sentences:
        for token in sentence:
            print(token, token.lemma_)
    if i==0: break

print("\nPreprocessing test --- END\n")


y = np.array([1, 1, 0, 1, 0])
X = [[1, 0, 1, 1],
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [1, 1, 1, 1]]
X = np.array(X)

model = BernoulliNaiveBayes()
model.train(X, y)
print(model.m)
print(model.count_y_1)
print(model.count_x_1_y_1)
print(model.count_x_1_y_0)

print(model.p_y(1), model.p_y(0))
print(model.p_xi_given_y([0, 0, 0, 0], 0, 0), model.p_xi_given_y([0, 0, 0, 0], 0, 1))

print(math.exp(model.log_prob_y_given_x([0, 0, 0, 0], 0)), math.exp(model.log_prob_y_given_x([0, 0, 0, 0], 1)))
print(model.predict_class([0, 0, 0, 0]))

print(math.exp(model.log_prob_y_given_x([1, 0, 0, 1], 0)), math.exp(model.log_prob_y_given_x([1, 0, 0, 1], 1)))
print(model.predict_class([1, 0, 0, 1]))