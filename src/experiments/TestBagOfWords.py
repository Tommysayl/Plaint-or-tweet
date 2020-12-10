import math, time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import binarize
from src.models.BernoulliNaiveBayes import BernoulliNaiveBayes
from src.datasets.TwitterDSReader import TwitterDSReader

def preprocessing(self, ds):
    ds.read_from_file()
    for text in ds.docs():
        new = ''
        for token in text.tokens: new = new + ' ' + token.lemma_
        self.corpus += [new]
        self.y += [text.label]
    return self.corpus, self.y

def save_preprocessing(self, path, ds):
    self.preprocessing(ds)
    df = pd.DataFrame(self.corpus)
    df.insert(0, 'Label', self.y, True)
    #df = df.astype(np.uint8)
    df.to_csv(path)

def load_preprocessing(path):
    df = pd.read_csv(path)
    corpus = df.iloc[:,2].values
    y = df['Label'].values
    return corpus, y

if __name__ == '__main__':
    start_time = time.time()

    SEED = 42
    TRAIN_PERC = 0.8

    X, y = load_preprocessing('bow_preprocess.csv')
    y = y // 4 #labels in {0, 1}
    print('preprocessing done')

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_PERC, random_state=SEED)
    print('train:', X_train.shape)
    print('test:', X_test.shape)
    
    vectorizer = CountVectorizer(ngram_range=(1,1)) #(2,2) for bigrams and so on
    vectorizer.fit(X_train.astype('str'))
    print('vectorizer trained')
    #dictionary = set(vectorizer.get_feature_names())
    #print('dictionary size:', len(dictionary))
    
    X_train_bow = vectorizer.transform(X_train.astype('str'))
    binarize(X_train_bow, copy=False)
    print('train data vectorized')
    model = BernoulliNaiveBayes()
    model.train(X_train_bow, y_train)
    print('model trained')

    print(model.m)
    print(model.count_y_1)
    print(model.count_x_1_y_0)
    print(model.count_x_1_y_1)

    print('seconds needed:', (time.time() - start_time))