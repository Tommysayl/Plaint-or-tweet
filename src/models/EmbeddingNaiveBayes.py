from src.datasets.Embedder import Embedder
import logging
import numpy as np
from src.utils import discretizeVector
from src.models.CategoricalNaiveBayes import CategoricalNaiveBayes
from src.models.MultinomialNaiveBayes import MultinomialNaiveBayes

class EmbeddingNaiveBayes():
    def __init__(self, classifier='categorical', fastText=True, embeddingSize = 100, numBinsPerFeature = 10, loadEmbedderPath = None, exportEmbedderPath = None):
        '''fastText=False ==> word2vec'''
        assert classifier in ['categorical', 'multinomial']  #classifier can be categorical, gaussian or multinomial
        self.numBinsPerFeature = [numBinsPerFeature] * embeddingSize #assume same number of bins for each feature
        self.threshold = 0.5
        self.classifierType = classifier
        self.fastText = fastText
        self.embeddingSize = embeddingSize
        self.embedder = None
        self.load_embedder = loadEmbedderPath
        self.export_embedder = exportEmbedderPath
        self.model = None

        self.minMaxPair = None #min/max pair used for categorical classifier
        self.minPerFeature = None #minimum for each column (used in multinomial)

    def train(self, X_train, y_train, silent = False):
        '''train the model, X_train contains the tweet in each row'''

        #====== prepare embedder
        if not silent:
            logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO) #gensim logging
        self.embedder = Embedder()
        if self.fastText: #train fast text embedder
            self.embedder.train_ft(X_train, self.embeddingSize, load_path=self.load_embedder)
        else: #train word2vec embedder
            self.embedder.train_w2v(X_train, self.embeddingSize, load_path=self.load_embedder)
        
        if self.export_embedder is not None:
            self.embedder.save_model(self.export_embedder)

        if not silent:
            print('embedder trained')

        X_train_vec = self.embedder.sentence_embedding(X_train)
        if not silent:
            print('train embeddings computed')

        #===== build model

        #categorical classifier: we discretize all the features
        if self.classifierType == 'categorical':
            self.minMaxPair = np.apply_along_axis(lambda x: [min(x), max(x)], 0, X_train_vec) #find (min, max) for each column of X_train_vec
            X_train_vec = np.array([discretizeVector(v, self.minMaxPair[0][i], self.minMaxPair[1][i], self.numBinsPerFeature[i]) for i,v in enumerate(X_train_vec.T)]).T #discretize each column
            if not silent:
                print('train data discretized')
            self.model = CategoricalNaiveBayes(cat_nums=self.numBinsPerFeature)
        elif self.classifierType == 'multinomial':
            self.minPerFeature = np.apply_along_axis(lambda x: min(x), 0, X_train_vec) #find min for each column of X_train_vec
            X_train_vec = np.apply_along_axis(lambda x: x - self.minPerFeature, 1, X_train_vec) #make all values non-negative (gives problems with log)
            self.model = MultinomialNaiveBayes()
        else:
            assert False, 'classifier type not known'

        self.model.train(X_train_vec, y_train)
        if not silent:
            print('model trained')

    def perform_test(self, X_test, silent=False):
        X_test_vec = self.embedder.sentence_embedding(X_test)
        if not silent:
            print('test embeddings computed')
        
        if self.classifierType == 'categorical':
            #note: we use minMaxPair computed in training
            X_test_vec = np.array([discretizeVector(v, self.minMaxPair[0][i], self.minMaxPair[1][i], self.numBinsPerFeature[i]) for i,v in enumerate(X_test_vec.T)]).T #discretize each column
            if not silent:
                print('test data discretized')
        elif self.classifierType == 'multinomial':
            X_test_vec = np.apply_along_axis(lambda x: x - self.minPerFeature, 1, X_test_vec)
            X_test_vec[X_test_vec < 0] = 0
        else:
            assert False, 'classifier type not known'
        
        y_score = self.model.multi_prediction_score(X_test_vec)
        y_pred = self.model.multi_predict_class_from_score(y_score, threshold=self.threshold)
        return y_score, y_pred
