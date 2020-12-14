import math, time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import binarize
from sklearn.metrics import accuracy_score, f1_score
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
    
    X_train_bow = vectorizer.transform(X_train.astype('str'))
    binarize(X_train_bow, copy=False)
    print('train data vectorized')
    model = BernoulliNaiveBayes()
    model.train(X_train_bow, y_train)
    print('model trained')
    
    X_test_bow = vectorizer.transform(X_test.astype('str'))
    binarize(X_test_bow, copy=False)
    print('test data vectorized')
    y_pred = model.multi_predict_class(X_test_bow)
    print('accuracy:', accuracy_score(y_test, y_pred))
    print('f1-score:', f1_score(y_test, y_pred))

    print('seconds needed:', (time.time() - start_time))