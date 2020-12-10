import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from BernoulliNaiveBayes import BernoulliNaiveBayes
from TwitterDSReader import TwitterDSReader

class DatasetBagOfWords():
    def __init__(self):
        self.X = []
        self.y = []
        self.vocabulary = []

        self.corpus = []
        self.vectorizer = None

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

    def load_preprocessing(self, path):
        df = pd.read_csv(path)
        self.corpus = df.iloc[:,2].values
        self.y = df['Label'].values

    def bag_of_words(self, perc, start=0, ngram=(1,1)):
        # Be careful to not overlap 
        #assert ngram[0] == ngram[1], "Incostintent N-gram!" # without it still works but is strange
        #assert perc >= 0 and perc <= 1 , "Range must be whitin corpus' length"

        #n = math.ceil(perc*len(self.corpus))

        #assert start + n < len(self.corpus), "Slice bigger than dataset"

        vectorizer = CountVectorizer(ngram_range=ngram, dtype=np.uint8) #(2,2) for bigrams and so on

        self.X = np.asarray( vectorizer.fit_transform(self.corpus[start:perc]).todense() )
        self.vocabulary = vectorizer.vocabulary_
        print(vectorizer.vocabulary_)
        #for i in vectorizer.vocabulary: if not i in self.vocabulary: self.vocabulary += [i] 
        self.X = np.asarray( vectorizer.fit_transform(self.corpus[start+perc:perc+perc]).todense() )
        print(vectorizer.vocabulary_)

        self.y = np.array(self.y)

    def save_bag_of_words(self, path):
        df = pd.DataFrame(self.X, )
        df.insert(0, 'Label', self.y, True)
        df = df.astype(np.uint8)
        print(df)
        #df.to_csv(path)

    def load_bag_of_words(self, path):
        df = pd.read_csv(path, index_col=False)
        self.y = df['Label'].values
        self.X = df.iloc[:, 2:].values
        self.vocabulary = dict( (label, i) for i, label in enumerate(df.columns[2:]) )

    def train(self, perc, ngram=(1,1)):
        assert ngram[0] == ngram[1], "Incostintent N-gram!" # without it still works but is strange
        assert perc >= 0 and perc <= 1, "Wrong train percentage!"

        n = math.ceil(perc*len(self.corpus))
        self.y = np.asarray(self.y[:n])
        train = self.corpus[:n]
        test = self.corpus[n:]

        vectorizer = CountVectorizer(ngram_range=ngram, dtype=np.uint8) #(2,2) for bigrams and so on
        np.asarray(vectorizer.fit(train.astype('U')))
        self.vocabulary = vectorizer.vocabulary_

        vectorizer = CountVectorizer(ngram_range=ngram, dtype=np.uint8, vocabulary=self.vocabulary)

        self.X = np.asarray( vectorizer.transform(train.astype('U')).todense() )

        model = BernoulliNaiveBayes()
        model.train(self.X, self.y)

        print(model.m)
        print(model.count_y_1)
        print(model.count_x_1_y_1)
        print(model.count_x_1_y_0)
