from gensim.models import FastText 
from gensim.models import Word2Vec
import pandas as pd
import numpy as np


class Embedder():

    def __init__(self):
        self.corpus = []

    def load_corpus(self, csv_path):
        df = pd.read_csv(csv_path)
        self.corpus = df.iloc[:,2].values
        self.y = df['Label'].values
        return self.corpus
        

    def embed_ft (self, csv_path , size = 100, window = 5, min_count = 1, workers = 1, sg = 1):
        '''
        Input: Path to csv;
        size = embedding vector size;
        window = Maximum distance between the current and predicted word within a sentence
        min_count =  Ignores all words with total frequency lower than this.
        workers = threads to train model
        sg = 0 for CBOW, 1 for skip-gram
        Output: Gensim Model
        '''
        corpus = self.load_corpus(csv_path)
        word_corpus = [sentence.lower().split() for sentence in corpus if isinstance(sentence, str)]
        model = FastText(sentences = word_corpus, size = size, window = window, min_count = min_count, workers = workers, sg = sg)
        return model

    def embed_w2v (self, csv_path , size = 100, window = 5, min_count = 1, workers = 1, sg = 0):
        '''
        Input: Path to csv;
        size = embedding vector size;
        window = Maximum distance between the current and predicted word within a sentence
        min_count =  Ignores all words with total frequency lower than this.
        workers = threads to train model
        sg = 0 for CBOW, 1 for skip-gram
        Output: Gensim Model
        '''
        
        corpus = self.load_corpus(csv_path)
        word_corpus = [sentence.lower().split() for sentence in corpus if isinstance(sentence, str)]
        model = Word2Vec(sentences = word_corpus, size = size, window = window, min_count = min_count, workers = workers, sg = sg)
        return model  
        
    def discretize_vectors (self, model, n_ranges = 10):
        '''
        Input: FastText/Word2Vec model (created with self.embed_ft/w2v), int representing the number of ranges
        we want to discretize our values to. 
        Output; Dictionary where Keys = words, values = discretized embedded vectors
        '''
        support_dict = {}
        vector_dict = {}
        for word in model.wv.vocab:
            support_dict[word] = model.wv[word]

        
        for word in support_dict:
            vector_dict[word] = list()
            min_a, max_a = self.find_min_max(support_dict[word])
            for value in support_dict[word]:
                discretized = self.discretize_value(max_a, min_a, value, n_ranges)
                vector_dict[word].append(discretized)
        return vector_dict


    def discretize_value(self, max_a, min_a, value, n_ranges):
        normalized = (value - min_a) / (max_a - min_a)
        discretized = np.round(normalized * n_ranges) 
        return discretized

    def find_min_max(self, array):
        max_a = max(array)
        min_a = min(array)
        return max_a, min_a
        





        
        
        



    
