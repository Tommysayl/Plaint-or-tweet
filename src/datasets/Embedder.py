from gensim.models import FastText 
from gensim.models import Word2Vec
import pandas as pd
import numpy as np


class Embedder():

    def __init__(self):
        self.model = None

    def train_ft (self, corpus, size = 100, window = 5, min_count = 1, workers = 1, sg = 1, load_path = None):
        '''
        Input: Corpus to embed;
        size = embedding vector size;
        window = Maximum distance between the current and predicted word within a sentence
        min_count =  Ignores all words with total frequency lower than this.
        workers = threads to train model
        sg = 0 for CBOW, 1 for skip-gram
        Output: Gensim Model
        '''
        if load_path is None:
            word_corpus = [sentence.lower().split() for sentence in corpus if isinstance(sentence, str)]
            self.model = FastText(sentences = word_corpus, size = size, window = window, min_count = min_count, workers = workers, sg = sg)
        else:
            self.model = FastText.load(load_path)
        return self.model

    def train_w2v (self, csv_path , size = 100, window = 5, min_count = 1, workers = 1, sg = 1):
        '''
        Input: Corpus to embed;
        size = embedding vector size;
        window = Maximum distance between the current and predicted word within a sentence
        min_count =  Ignores all words with total frequency lower than this.
        workers = threads to train model
        sg = 0 for CBOW, 1 for skip-gram
        Output: Gensim Model
        '''
        word_corpus = [sentence.lower().split() for sentence in corpus if isinstance(sentence, str)]
        self.model = Word2Vec(sentences = word_corpus, size = size, window = window, min_count = min_count, workers = workers, sg = sg)
        return self.model  

    def avarage_vector(self, vects):
        assert vects.size > 0
        mean_v = np.mean(vects, axis=0)
        return mean_v


    def sentence_embedding(self, corpus):
        '''
        computes sentence embeddings
        Input: corpus is a list of sentences
        Output: numpy array, i-th row is embedding of sentence i
        '''
        sents = [[self.model.wv[word] if word in model.wv else np.zeros(100) for word in str(sent).lower().split()] for sent in corpus]
        embedded_sents = [self.avarage_vector(np.array(sent)) for sent in sents]
        return np.array(embedded_sents)