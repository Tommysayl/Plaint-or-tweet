from src.models.BernoulliNaiveBayes import BernoulliNaiveBayes
from src.models.MultinomialNaiveBayes import MultinomialNaiveBayes
from sklearn.preprocessing import binarize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, accuracy_score
from src.utils import getBestThreshold
import numpy as np

class BagOfWordsNaiveBayes():
    def __init__(self, multinomial, useTfIdf, ngram_start, ngram_end):
        '''multinomial = True means that we want to count occurrences of the words, ie: use multinomial'''

        assert multinomial or (not useTfIdf) #can't use tfidf without multinomial

        self.useTfIdf = useTfIdf
        self.ngram_s = ngram_start
        self.ngram_e = ngram_end
        self.multinomial = multinomial
        self.threshold = 0.5
        self.vectorizer = None
        self.model = None

    def train(self, X_train, y_train, silent = False):
        '''train the model, X_train contains the tweet in each row'''
        if self.useTfIdf:
            self.vectorizer = TfidfVectorizer(ngram_range=(self.ngram_s, self.ngram_e), tokenizer=lambda x: x.split(), lowercase=False, preprocessor=lambda x: x)
        else:
            self.vectorizer = CountVectorizer(ngram_range=(self.ngram_s, self.ngram_e), tokenizer=lambda x: x.split(), lowercase=False, preprocessor=lambda x: x)

        if self.multinomial:
            self.model = MultinomialNaiveBayes()
        else:
            self.model = BernoulliNaiveBayes()

        self.vectorizer.fit(X_train.astype('str'))
        #assert len(self.vectorizer.stop_words_) == 0 #we don't want preprocess by scikit learn, we already performed it
        #print(self.vectorizer.get_feature_names())
        
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

    def cross_validation(self, X_train, y_train, X_val, y_val, silent = False):
        ns = self.ngram_s
        ne = self.ngram_e
        #try all possibilities of ngrams in (start, end)
        bestAcc = -1
        bestNgram = (0, 0)
        bestYscore = None
        for i in range(ns, ne+1): 
            for j in range(i, ne+1):
                if not silent:
                    print(f'trying ngram ({i},{j})')
                self.ngram_s = i
                self.ngram_e = j
                self.train(X_train, y_train)
                y_score, y_pred = self.perform_test(X_val)
                acc = accuracy_score(y_val, y_pred)
                if not silent:
                    print('ngram accuracy:', acc)
                if acc > bestAcc:
                    bestAcc = acc
                    bestNgram = (i, j)
                    bestYscore = y_score
        
        if not silent:
            print(f'best ngram:{bestNgram[0], bestNgram[1]}')
        self.ngram_s = bestNgram[0]
        self.ngram_e = bestNgram[1]

        #find best threshold
        if bestYscore is None:
            self.train(X_train, y_train)
            bestYscore, _ = self.perform_test(X_val)
        fpr, tpr, thresholds = roc_curve(y_val, bestYscore)
        self.threshold = getBestThreshold(tpr, fpr, thresholds)
        if not silent:
            print('best threshold:', self.threshold)

    def to_dict(self):
        return self.model.to_dict()