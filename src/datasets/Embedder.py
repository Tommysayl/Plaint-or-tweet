from gensim.models import FastText 
from gensim.models import Word2Vec
import pandas as pd


class Embedder():

    def __init__(self):
        self.corpus = []

    def load_corpus(self, csv_path):
        df = pd.read_csv(csv_path)
        self.corpus = df.iloc[:,2].values
        self.y = df['Label'].values
        return self.corpus
        

    def embed_ft (self, csv_path , size, window, min_count, workers, sg):
        corpus = self.load_corpus(csv_path)
        word_corpus = [sentence.lower().split() for sentence in corpus if isinstance(sentence, str)]
        model = FastText(sentences = word_corpus, size = size, window = window, min_count = min_count, workers = workers, sg = sg)
        return model

    def embed_w2v (self,csv_path, size, window, min_count, workers, sg):
        corpus = self.load_corpus(csv_path)
        word_corpus = [sentence.lower().split() for sentence in corpus if isinstance(sentence, str)]
        model = Word2Vec(sentences = word_corpus, size = size, window = window, min_count = min_count, workers = workers, sg = sg)
        return model  
        


    
