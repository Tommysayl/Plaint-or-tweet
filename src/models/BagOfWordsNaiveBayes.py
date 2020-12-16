from src.models.BernoulliNaiveBayes import BernoulliNaiveBayes
from src.models.MultinomialNaiveBayesSparse import MultinomialNaiveBayes
from sklearn.preprocessing import binarize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class BagOfWordsNaiveBayes():
    def __init__(self, multinomial, useTfIdf, ngram_start, ngram_end):
        '''multinomial = True means that we want to count occurrences of the words, ie: use multinomial'''

        assert multinomial or (not useTfIdf) #can't use tfidf without multinomial

        self.multinomial = multinomial

        if useTfIdf:
            self.vectorizer = TfidfVectorizer(ngram_range=(ngram_start, ngram_end))
        else:
            self.vectorizer = CountVectorizer(ngram_range=(ngram_start, ngram_end))

        if multinomial:
            self.model = MultinomialNaiveBayes()
        else:
            self.model = BernoulliNaiveBayes()

    def train(self, X_train, y_train):
        '''train the model, X_train contains the tweet in each row'''
       
        self.vectorizer.fit(X_train.astype('str'))
        print('vectorizer trained')
        
        X_train_bow = self.vectorizer.transform(X_train.astype('str'))
        if not self.multinomial:
            binarize(X_train_bow, copy=False)
        print('train data vectorized')
        
        self.model.train(X_train_bow, y_train)
        print('model trained')

    def perform_test(self, X_test, y_test):
        X_test_bow = self.vectorizer.transform(X_test.astype('str'))
        if not self.multinomial:
            binarize(X_test_bow, copy=False)
        print('test data vectorized')
        
        y_score = self.model.multi_prediction_score(X_test_bow)
        y_pred = self.model.multi_predict_class_from_score(y_score, 0.5)
        return y_score, y_pred

