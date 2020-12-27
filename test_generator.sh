#!/bin/sh

#prepare datasets
#we assume to have in projectFolder/datasets/ the files:
# - training.1600000.processed.noemoticon.csv
# - imdb.csv (merged files from the imdb download folder)
# - reddit.csv
#and we assume there exists the folder projectFolder/datasets/preprocess
python -m src.datasets.ComputePreprocessedDataset --dataset='twitter' --save_path='datasets/preprocess/twitter_preprocessed.csv'
python -m src.datasets.ComputePreprocessedDataset --dataset='twitter' --save_path='datasets/preprocess/twitter_not_preprocessed.csv' --preprocess=False
python -m src.datasets.ComputePreprocessedDataset --dataset='imdb' --save_path='datasets/preprocess/imdb_preprocessed.csv'
python -m src.datasets.ComputePreprocessedDataset --dataset='imdb' --save_path='datasets/preprocess/imdb_not_preprocessed.csv' --preprocess=False
python -m src.datasets.ComputePreprocessedDataset --dataset='reddit' --save_path='datasets/preprocess/reddit_preprocessed.csv'
python -m src.datasets.ComputePreprocessedDataset --dataset='reddit' --save_path='datasets/preprocess/reddit_not_preprocessed.csv' --preprocess=False

#test with and without preprocessing:
python -m src.experiments.TestSingleSplit --class1=4 --name='BernoulliBOW_NotPreprocessed' --preprocessing_path='datasets/preprocess/twitter_not_preprocessed.csv' --show=False
python -m src.experiments.TestSingleSplit --class1=4 --name='BernoulliBOW_Preprocessed' --preprocessing_path='datasets/preprocess/twitter_preprocessed.csv' --show=False
python -m src.experiments.TestSingleSplit --class1=4 --bow=False --classifierType='multinomial' --name='EmbeddingsMultinomial_NotPreprocessed' --preprocessing_path='datasets/preprocess/twitter_not_preprocessed.csv' --emb_import_path=None --emb_export_path='datasets/fasttext/twitter_embedding.ft' --show=False
python -m src.experiments.TestSingleSplit --class1=4 --bow=False --classifierType='multinomial' --name='EmbeddingsMultinomial_Preprocessed' --preprocessing_path='datasets/preprocess/twitter_preprocessed.csv' --emb_import_path=None --emb_export_path='datasets/fasttext/twitter_embedding_prep.ft' --show=False

#classic nb vs embeddings (add more seeds here)
python -m src.experiments.TestSingleSplit --class1=4 --preprocessing_path='datasets/preprocess/twitter_not_preprocessed.csv' --name='TwitterBernoulliBOW' --seed=42 --show=False
python -m src.experiments.TestSingleSplit --class1=4 --multinomial=True --preprocessing_path='datasets/preprocess/twitter_not_preprocessed.csv' --name='TwitterMultinomialBOW' --seed=42 --show=False
python -m src.experiments.TestSingleSplit --class1=4 --multinomial=True --tfidf=True --preprocessing_path='datasets/preprocess/twitter_not_preprocessed.csv' --name='TwitterMultinomialTfIdfBOW' --seed=42 --show=False
python -m src.experiments.TestSingleSplit --class1=4 --bow=False --preprocessing_path='datasets/preprocess/twitter_not_preprocessed.csv' --classifierType='multinomial' --emb_import_path='datasets/fasttext/twitter_embedding.ft' --name='TwitterMultinomialFastText' --seed=42 --show=False
python -m src.experiments.TestSingleSplit --class1=4 --bow=False --preprocessing_path='datasets/preprocess/twitter_not_preprocessed.csv' --classifierType='categorical' --emb_import_path='datasets/fasttext/twitter_embedding.ft' --name='TwitterCategoricalFastText' --seed=42 --show=False
python -m src.experiments.TestSingleSplit --class1=4 --bow=False --preprocessing_path='datasets/preprocess/twitter_not_preprocessed.csv' --classifierType='gaussian' --emb_import_path='datasets/fasttext/twitter_embedding.ft' --name='TwitterGaussianFastText' --seed=42 --show=False

#find best model
python -m src.experiments.TestSingleSplit --class1=4 --train_perc=0.6 --validation_perc=0.2 --ngram_e=3 --preprocessing_path='datasets/preprocess/twitter_not_preprocessed.csv' --name='TwitterBernoulliBOW_CROSS' --seed=42 --show=False
python -m src.experiments.TestSingleSplit --class1=4 --train_perc=0.6 --validation_perc=0.2 --ngram_e=3 --multinomial=True --preprocessing_path='datasets/preprocess/twitter_not_preprocessed.csv' --name='TwitterMultinomialBOW_CROSS' --seed=42 --show=False
python -m src.experiments.TestSingleSplit --class1=4 --train_perc=0.6 --validation_perc=0.2 --ngram_e=3 --multinomial=True --tfidf=True --preprocessing_path='datasets/preprocess/twitter_not_preprocessed.csv' --name='TwitterMultinomialTfIdfBOW_CROSS' --seed=42 --show=False

#test best model (multinomial tf-idf with (1,3)-gram) on 0.8/0.2
python -m src.experiments.TestSingleSplit --class1=4 --train_perc=0.8 --ngram_e=3 --multinomial=True --tfidf=True --preprocessing_path='datasets/preprocess/twitter_not_preprocessed.csv' --name='TwitterMultinomialTfIdfBOW(1,3)' --seed=42 --show=False


#======== unusual tests: train on one dataset, test on another one

#base results [we use bernoulli since it's faster] (train and test on same dataset)
#imdb 
python -m src.experiments.TestSingleSplit --class1=1 --ngram_e=3 --preprocessing_path='datasets/preprocess/imdb_not_preprocessed.csv' --name='IMDBBernoulliBOW(1,3)' --seed=42 --show=False
#reddit
python -m src.experiments.TestSingleSplit --class1=1 --ngram_e=3 --preprocessing_path='datasets/preprocess/reddit_not_preprocessed.csv' --name='RedditBernoulliBOW(1,3)' --seed=42 --show=False
#train on imdb and test on twitter, and viceversa: train on twitter and test on imdb
python -m src.experiments.ImdbTwitterSingleSplit --ngram_e=3 --path_imdb='datasets/preprocess/imdb_not_preprocessed.csv' --path_tweet='datasets/preprocess/twitter_not_preprocessed.csv' --name='_ImdbVsTwitter_BernoulliBOW(1,3)' --seed=42 --show=False
#train on reddit and test on twitter, and viceversa: train on twitter and test on reddit
python -m src.experiments.ImdbTwitterSingleSplit --ngram_e=3 --path_imdb='datasets/preprocess/reddit_not_preprocessed.csv' --path_tweet='datasets/preprocess/twitter_not_preprocessed.csv' --name='_RedditVsTwitter_BernoulliBOW(1,3)' --seed=42 --show=False
