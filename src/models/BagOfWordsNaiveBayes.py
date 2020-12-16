from src.models.BernoulliNaiveBayes import BernoulliNaiveBayes
from src.models.MultinomialNaiveBayesSparse import MultinomialNaiveBayes
from sklearn.preprocessing import binarize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from src.utils import getBestThreshold

class BagOfWordsNaiveBayes():
    def __init__(self, multinomial, useTfIdf, ngram_start, ngram_end):
        '''multinomial = True means that we want to count occurrences of the words, ie: use multinomial'''

        assert multinomial or (not useTfIdf) #can't use tfidf without multinomial

        self.multinomial = multinomial
        self.threshold = 0.5

        if useTfIdf:
            self.vectorizer = TfidfVectorizer(ngram_range=(ngram_start, ngram_end))
        else:
            self.vectorizer = CountVectorizer(ngram_range=(ngram_start, ngram_end))

        if multinomial:
            self.model = MultinomialNaiveBayes()
        else:
            self.model = BernoulliNaiveBayes()

    def train(self, X_train, y_train, silent = False):
        '''train the model, X_train contains the tweet in each row'''
       
        self.vectorizer.fit(X_train.astype('str'))
        if not silent:
            print('vectorizer trained')
        
        X_train_bow = self.vectorizer.transform(X_train.astype('str'))
        if not self.multinomial:
            binarize(X_train_bow, copy=False)
        if not silent:
            print('train data vectorized')
        
        self.model.train(X_train_bow, y_train)
        if not silent:
            print('model trained')

    def perform_test(self, X_test, silent=False):
        X_test_bow = self.vectorizer.transform(X_test.astype('str'))
        if not self.multinomial:
            binarize(X_test_bow, copy=False)
        if not silent:
            print('test data vectorized')
        
        y_score = self.model.multi_prediction_score(X_test_bow)
        y_pred = self.model.multi_predict_class_from_score(y_score, threshold=self.threshold)
        return y_score, y_pred

    def kFoldBestThresholdSearch(self, X_train, y_train, seed, splits = 3):
        kf = KFold(n_splits = splits, random_state=seed, shuffle=True)
        bestThresholds = []
        i = 0
        for train_idx, val_idx in kf.split(X_train):
            print('begin iteration', i)
            i+=1
            X_train_small, X_val = X_train[train_idx], X_train[val_idx]
            y_train_small, y_val = y_train[train_idx], y_train[val_idx]
            self.model.reset_params()
            self.train(X_train_small, y_train_small, silent=True)
            y_score, y_pred = self.perform_test(X_val, silent=True)
            fpr, tpr, thresholds = roc_curve(y_val, y_score)
            bestThresholds += [getBestThreshold(tpr, fpr, thresholds)]

        self.model.reset_params()
        self.threshold = sum(bestThresholds) / len(bestThresholds)
        print('best thresholds:', bestThresholds)
        print('threshold:', self.threshold)