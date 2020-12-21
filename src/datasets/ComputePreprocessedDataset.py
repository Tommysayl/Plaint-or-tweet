'''
this script is used to preprocess a dataset and save it. 
This is useful because we don't need to perform the preprocessing of the dataset every time we run a test, which saves much time
'''
import fire, time
import numpy as np
import pandas as pd
from src.datasets.TwitterDSReader import TwitterDSReader
from src.datasets.ImdbDSReader import ImdbDSReader

def preprocessing(ds, path):
    ds.read_from_file(p=path)
    corpus = []
    y = []
    for text in ds.docs():
        new = ''
        for token in text.tokens: new = new + ' ' + token.lemma_
        corpus += [new]
        y += [text.label]
    return np.asarray(corpus), np.asarray(y)

def save_preprocessing(path, corpus, y):
    df = pd.DataFrame(corpus)
    df.insert(0, 'Label', y, True)
    #df = df.astype(np.uint8)
    df.to_csv(path)

def extractCorpusAndLabel(datasetName, path):
    if datasetName in {'twitter', 'twitter60k'}:
        df = pd.read_csv(path, encoding="ISO-8859-1", names=["label", "id", "date", "query", "username", "text"])
        return df["text"], df["label"]
    elif datasetName == 'imdb':
        df = pd.read_csv(path, encoding="ISO-8859-1", names=["text", "label"])
        return df["text"][1:], df["label"][1:].astype(np.int64)
    elif datasetName == 'nfl':
        df = pd.read_csv(path, encoding="ISO-8859-1", names=["id", "timestamp", "team", "subreddit", "sentiment", "text"])
        corpus = df["text"][1:]
        label = df["sentiment"][1:].astype(np.float64)
        mask = (label < -0.5) | (label > 0.5)  # Remove elements in [-0.5,0.5]
        label = list(map(lambda x: 0 if x < 0 else 1, label[mask])) # discretize elements in 0,1
        return corpus[mask], np.asarray(label, dtype=np.int64)
        
    
def main(dataset='twitter', preprocess=True, save_path = 'datasets/preprocess/twitter_preprocessed.csv'):
    assert dataset in {'twitter', 'twitter60k', 'imdb', 'nfl'} #supported datasets
    start_time = time.time()

    path = ''
    if dataset == 'twitter':
        path = 'datasets/training.1600000.processed.noemoticon.csv'
    elif dataset == 'twitter60k':
        path = 'datasets/twitter.csv'
    elif dataset == 'imdb':
        path = 'datasets/imdb.csv'
    elif dataset == 'nfl':
        path = 'datasets/nfl.csv'
    
    dsr = None
    if preprocess: #otherwise not needed
        if dataset in {'twitter', 'twitter60k'}:
            dsr = TwitterDSReader()
        elif dataset == 'imdb':
            dsr = ImdbDSReader()
        
    print('ds reader created')

    corpus, labels = None, None
    if preprocess:
        corpus, labels = preprocessing(dsr, path)
    else:
        corpus, labels = extractCorpusAndLabel(dataset, path)
    
    print('dataset preprocessed', (time.time() - start_time))

    save_preprocessing(save_path, corpus, labels)

    print('saved', (time.time() - start_time))

if __name__ == '__main__':
    fire.Fire(main)