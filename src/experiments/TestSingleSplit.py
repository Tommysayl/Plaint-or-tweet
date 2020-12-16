import math, time
import numpy as np
import pandas as pd
import fire
from src.models.BagOfWordsNaiveBayes import BagOfWordsNaiveBayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt

def preprocessing(ds):
    ds.read_from_file()
    corpus = []
    y = []
    for text in ds.docs():
        new = ''
        for token in text.tokens: new = new + ' ' + token.lemma_
        corpus += [new]
        y += [text.label]
    return corpus, y

def save_preprocessing(path, corpus, y):
    df = pd.DataFrame(corpus)
    df.insert(0, 'Label', y, True)
    #df = df.astype(np.uint8)
    df.to_csv(path)

def load_preprocessing(path):
    df = pd.read_csv(path)
    corpus = df.iloc[:,2].values
    y = df['Label'].values
    return corpus, y

def main(seed = 42, train_perc = 0.8, multinomial=False, tfidf=False, ngram_s=1, ngram_e=1, findBestThreshold=False, preprocessing_path = 'bow_preprocess.csv'):
    start_time = time.time()

    print('seed:', seed)
    print('train_perc:', train_perc)
    print('multinomial:', multinomial)
    print('tfidf:', tfidf)
    print('ngram=(', ngram_s, ',', ngram_e, ')', sep='')

    if preprocessing_path is None:
        X, y = preprocessing(TwitterDSReader())
    else:
        X, y = load_preprocessing('bow_preprocess.csv')
    y = y // 4 #labels in {0, 1}
    print('preprocessing done')

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_perc, random_state=seed) #split in train/test
    print('train:', X_train.shape)
    print('test:', X_test.shape)
    
    model = BagOfWordsNaiveBayes(multinomial, tfidf, ngram_s, ngram_e) #create the model
    if findBestThreshold: #if you want best threshold, perform kfold 
        model.kFoldBestThresholdSearch(X_train, y_train, seed, splits=3) 
    model.train(X_train, y_train) #train the model
    y_score, y_pred = model.perform_test(X_test) #get scores and predictions
    fpr, tpr, thresholds = roc_curve(y_test, y_score) 
    print('accuracy:', accuracy_score(y_test, y_pred)) #print some scores
    print('f1-score:', f1_score(y_test, y_pred))
    print('au-roc:', roc_auc_score(y_test, y_score))

    #we perform the evalutation over the training set (just to compare with the test)
    y_score_train, y_pred_train = model.perform_test(X_train)
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_score_train)
    print('accuracy:', accuracy_score(y_train, y_pred_train))
    print('f1-score:', f1_score(y_train, y_pred_train))
    print('au-roc:', roc_auc_score(y_train, y_score_train))

    print('seconds needed:', (time.time() - start_time))

    plt.plot(fpr, tpr, label='test roc')
    plt.plot(fpr_train, tpr_train, label='train roc')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    fire.Fire(main)