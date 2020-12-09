import numpy as np
import pandas as pd
import math
from sklearn.feature_extraction.text import CountVectorizer

class DatasetBagOfWords():
    def __init__(self):
        self.X = []
        self.y = []

        self.__corpus = []
        self.__vectorizer = None

    def preprocessing(self, ds):
        ds.read_from_file()
        for i, text in enumerate(ds.docs()):
            new = ''
            for token in text.tokens: new = new + ' ' + token.lemma_
            self.__corpus += [new]
            self.y += [text.label]
            if i==10: break #remove

    def bag_of_words(self, ds, ngram=(1,1)):
        assert ngram[0] == ngram[1], "Incostintent N-gram!" # without it still works but is strange

        self.preprocessing(ds)

        self.__vectorizer = CountVectorizer(ngram_range=ngram, dtype=np.uint8) #(2,2) for bigrams
        self.X = np.asarray( self.__vectorizer.fit_transform(self.__corpus).todense() )
        self.y = np.array(self.y)

    def embed(self, text, ngram=(1,1)):
        vectorizer = CountVectorizer(ngram_range=ngram, dtype=np.uint8)
        return np.asarray( vectorizer.fit_transform(self.__corpus).todense() )

    def save(self, path):
        df = pd.DataFrame(self.X, columns=self.__vectorizer.get_feature_names())
        df.insert(0, 'Label', self.y, True)
        df = df.astype(np.uint8)
        df.to_csv(path)
        print(self.__vectorizer.vocabulary_)
        print(df.index)

    def load(self, path):
        df = pd.read_csv(path, index_col=False)
        self.y = df['Label'].values
        self.X = df.iloc[:, 2:].values
        return df