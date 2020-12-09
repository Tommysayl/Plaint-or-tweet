import numpy as np
import pandas as pd
from BernoulliNaiveBayes import BernoulliNaiveBayes
import math
from TwitterDSReader import TwitterDSReader
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

print("Preprocessing test --- START")
corpus = []
y = []

ds = TwitterDSReader()
ds.read_from_file()
for i, tweet in enumerate(ds.docs()):
    new = ''
    for token in tweet.tokens: new = new + ' ' + token.lemma_
    corpus += [new]
    y += [tweet.label]
    if i==10: break

print("\nPreprocessing test --- END")

# Bag of words

vectorizer = CountVectorizer(ngram_range=(1,1)) #(2,2) for bigrams
X = np.asarray(
    vectorizer
    .fit_transform(corpus)
    .todense())

y = np.array(y)

model = BernoulliNaiveBayes()
model.train(X, y)