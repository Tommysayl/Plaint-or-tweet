import math, time, json
import numpy as np
import pandas as pd
import fire
from src.models.BagOfWordsNaiveBayes import BagOfWordsNaiveBayes
from src.models.EmbeddingNaiveBayes import EmbeddingNaiveBayes
from src.datasets.TwitterDSReader import TwitterDSReader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
import pickle

def load_preprocessing(path):
    df = pd.read_csv(path)
    corpus = df.iloc[:,2].values
    y = df['Label'].values
    return corpus, y

def main(name='', class1=4, seed = 42, train_perc = 0.8, validation_perc=None, bow=True, 
multinomial=False, tfidf=False, ngram_s=1, ngram_e=1, 
fastText=True, classifierType = 'categorical', numBinsPerFeature=10, embeddingSize = 100, emb_export_path = None, emb_import_path = 'datasets/fasttext/train_embedding.ft', 
showTrainingStats=False, export_results_path='experiments/testSingleSplit', preprocessing_path = 'datasets/preprocess/twitter_preprocessed.csv',
export_model=False, export_model_path="export/"):
    '''
    if validation_perc is None, we don't do cross validation to find hyperparameters

    bow=True --> use bag of words, bow=False --> use embeddings
    - multinomial, tfidf, ngram_s, ngram_e ==> used only in Bag of Words
    - fastText, classifierType, numBinsPerFeature, embeddingSize ==> used only with embeddings
    '''
    start_time = time.time()

    print('seed:', seed)
    print('train_perc:', train_perc)
    if bow:
        print('multinomial:', multinomial)
        print('tfidf:', tfidf)
        print('ngram=(', ngram_s, ',', ngram_e, ')', sep='')
    else:
        print('fastText:', fastText)
        print('classifierType:', classifierType)
        print('embeddingSize:', embeddingSize)
        print('numBinsPerFeature:', numBinsPerFeature)

    X, y = load_preprocessing(preprocessing_path)
    y = y // class1 #labels in {0, 1}
    print('preprocessing loaded')

    print(X[:5])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_perc, random_state=seed) #split in train/test
    X_val, y_val = None, None
    if validation_perc is not None:
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=validation_perc/(1-train_perc), random_state=seed+1)
    print('train:', X_train.shape)
    print('test:', X_test.shape)
    if validation_perc is not None:
        print('validation:', X_val.shape)
    
    #create the model
    model = None 
    if bow: #bag of words model
        model = BagOfWordsNaiveBayes(multinomial, tfidf, ngram_s, ngram_e) #create the model
    else: #embeddings model
        model = EmbeddingNaiveBayes(classifierType, fastText, embeddingSize, numBinsPerFeature, loadEmbedderPath=emb_import_path, exportEmbedderPath=emb_export_path)

    if validation_perc is not None:
        model.cross_validation(X_train, y_train, X_val, y_val)

    model.train(X_train, y_train) #train the model
    y_score, y_pred = model.perform_test(X_test) #get scores and predictions
    fpr, tpr, thresholds = roc_curve(y_test, y_score) 
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_auroc = roc_auc_score(y_test, y_score)
    print('accuracy:', test_acc) #print some scores
    print('f1-score:', test_f1)
    print('au-roc:', test_auroc)

    #we perform the evalutation over the training set (just to compare with the test)
    if showTrainingStats:
        y_score_train, y_pred_train = model.perform_test(X_train)
        fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_score_train)
        print('accuracy:', accuracy_score(y_train, y_pred_train))
        print('f1-score:', f1_score(y_train, y_pred_train))
        print('au-roc:', roc_auc_score(y_train, y_score_train))

    print('seconds needed:', (time.time() - start_time))

    bestngram = None
    if bow:
        bestngram = (model.ngram_s, model.ngram_e)

    exportStats(export_results_path, name, seed, train_perc, validation_perc, bow, multinomial, tfidf, ngram_s, ngram_e, 
    fastText, classifierType, embeddingSize, numBinsPerFeature, test_acc, test_f1, test_auroc, fpr, tpr, preprocessing_path, bestngram, model.threshold)

    if export_model:
        export_ml_model(name, model, export_model_path)

    plt.plot(fpr, tpr, label='test roc')
    if showTrainingStats:
        plt.plot(fpr_train, tpr_train, label='train roc')
    plt.legend()
    plt.show()

def exportStats(path, name, seed, train_perc, val_perc, bow, multinomial, tfidf, ngram_s, ngram_e, 
fastText, classifierType, embeddingSize, numBinsPerFeature, accuracy, f1, auroc, fpr, tpr, preprocessing_path, bestngram, bestthreshold):
    if path is None:
        return
    path += '_'+name+'_'+str(time.time())
    outd = {'name': name,
            'seed': seed, 
            'train_perc': train_perc,
            'val_perc': val_perc,
            'bow': bow,
            'accuracy': accuracy,
            'f1-score': f1,
            'auroc': auroc,
            'preprocessing_path': preprocessing_path,
            'best_threshold': bestthreshold,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()}
    if bow:
        outd['multinomial'] = multinomial
        outd['tfidf'] = tfidf 
        outd['ngram_s'] = ngram_s
        outd['ngram_e'] = ngram_e
        outd['bestngram_s'] = bestngram[0]
        outd['bestngram_e'] = bestngram[1]
    else:
        outd['fasttext'] = fastText
        outd['classifierType'] = classifierType
        outd['embeddingSize'] = embeddingSize
        outd['numBinsPeFeature'] = numBinsPerFeature
    
    with open(path, 'w') as fout:
        fout.write(json.dumps(outd))

# Exports the model to a file
def export_ml_model(name, model, path):
    path += name if name != "" else str(time.time())
    path += ".potmodel"
    print("Exporting model")
    obj = model.to_dict()
    with open(path, 'wb') as fout:
        fout.write(pickle.dumps(obj))
    print("Model exported to " + path)

if __name__ == '__main__':
    fire.Fire(main)