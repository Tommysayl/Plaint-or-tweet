import math, time
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import binarize
from sklearn.metrics import accuracy_score, f1_score
from src.models.CategoricalNaiveBayes import CategoricalNaiveBayes
from src.datasets.TwitterDSReader import TwitterDSReader
from src.datasets.Embedder import Embedder
from src.utils import discretizeVector

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

    class1 = 4 # 1 for Imdb, 4 for Twitter
    INPUT_PATH = 'twitter_preprocess.csv'
    OUTPUT_EMBEDDER = 'datasets/fasttext/train_embedding.ft'
    LOAD_EMBEDDER = None #'datasets/fasttext/train_embedding.ft'

    SEED = 42
    TRAIN_PERC = 0.8
    
    numFeaturesEmbedding = 100
    numBinsPerFeature = [20] * numFeaturesEmbedding

    X, y = load_preprocessing(INPUT_PATH)
    if class1 == 1: X = X[1:]
    y = y[1:].astype(np.int64) // class1 if class1 == 1 else y // class1 // class1 #labels in {0, 1}
    print('preprocessing done', (time.time() - start_time))

    X_tmp = []
    y_tmp = []
    for i in range(len(X)):
        if len(str(X[i]).split()) > 0:
            X_tmp.append(X[i])
            y_tmp.append(y[i])
    X = np.array(X_tmp)
    y = np.array(y_tmp)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_PERC, random_state=SEED)
    print('train:', X_train.shape, (time.time() - start_time))
    print('test:', X_test.shape, (time.time() - start_time))

    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO) #gensim logging
    embedder = Embedder()
    embedder.train_ft(X_train, size = numFeaturesEmbedding, load_path=LOAD_EMBEDDER)
    if OUTPUT_EMBEDDER is not None:
        embedder.model.save(OUTPUT_EMBEDDER)
    print('embedder trained', (time.time() - start_time))
    
    X_train_vec = embedder.sentence_embedding(X_train)
    minMaxPair = np.apply_along_axis(lambda x: [min(x), max(x)], 0, X_train_vec) #find (min, max) for each column of X_train_vec
    X_train_vec = np.array([discretizeVector(v, minMaxPair[0][i], minMaxPair[1][i], numBinsPerFeature[i]) for i,v in enumerate(X_train_vec.T)]).T #discretize each column
    print('train embeddings computed', (time.time() - start_time))
    model = CategoricalNaiveBayes()
    model.train(X_train_vec, y_train)
    print('model trained', (time.time() - start_time))

    X_test_vec = embedder.sentence_embedding(X_test)
    #note: we use minMaxPair computed in training
    X_test_vec = np.array([discretizeVector(v, minMaxPair[0][i], minMaxPair[1][i], numBinsPerFeature[i]) for i,v in enumerate(X_test_vec.T)]).T #discretize each column
    print('test embeddings computed', (time.time() - start_time))
    y_pred = model.multi_predict_class(X_test_vec)
    print('accuracy:', accuracy_score(y_test, y_pred))
    print('f1-score:', f1_score(y_test, y_pred))

    print('seconds needed:', (time.time() - start_time))