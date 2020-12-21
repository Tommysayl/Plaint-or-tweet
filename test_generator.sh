#!/bin/sh

# Twitter

# BOW
python -m src.experiments.TestSingleSplit --class1=4 --ngram_e=3 --preprocessing_path='datasets/preprocess/twitter_not_preprocessed.csv' --classifierType='bernoulli' --name='TwitterBernoulliBOW(1,3)105' --seed=105 --show=False
echo
python -m src.experiments.TestSingleSplit --class1=4 --ngram_e=3 --preprocessing_path='datasets/preprocess/twitter_not_preprocessed.csv' --classifierType='multinomial' --multinomial=True --name='TwitterMultinomialBOW(1,3)105' --seed=105 --show=False
echo
# Tfidf
python -m src.experiments.TestSingleSplit --class1=4 --ngram_e=3 --preprocessing_path='datasets/preprocess/twitter_not_preprocessed.csv' --classifierType='multinomial' --multinomial=True --tfidf=True --name='TwitterMultinomialTfidfBOW(1,3)105' --seed=105 --show=False
echo
# Fasttext
python -m src.experiments.TestSingleSplit --class1=4 --bow=False --ngram_e=3 --preprocessing_path='datasets/preprocess/twitter_not_preprocessed.csv' --classifierType='multinomial' --multinomial=True --emb_import_path='datasets/fasttext/twitter_embedding.ft' --name='TwitterMultinomialFastText(1,3)105' --seed=105 --show=False
echo
python -m src.experiments.TestSingleSplit --class1=4 --bow=False --ngram_e=3 --preprocessing_path='datasets/preprocess/twitter_not_preprocessed.csv' --classifierType='multinomial' --multinomial=True --emb_import_path='datasets/fasttext/twitter_embedding.ft' --tfidf=True --name='TwitterMultinomialTfidfFastText(1,3)105' --seed=105 --show=False
echo
python -m src.experiments.TestSingleSplit --class1=4 --bow=False --ngram_e=3 --preprocessing_path='datasets/preprocess/twitter_not_preprocessed.csv' --classifierType='gaussian' --emb_import_path='datasets/fasttext/twitter_embedding.ft' --name='TwitterGaussianFastText(1,3)105' --seed=105 --show=False
echo
python -m src.experiments.TestSingleSplit --class1=4 --bow=False --ngram_e=3 --preprocessing_path='datasets/preprocess/twitter_not_preprocessed.csv' --classifierType='categorical' --emb_import_path='datasets/fasttext/twitter_embedding.ft' --name='TwitterCategoricalFastText(1,3)105' --seed=105 --show=False
echo

# IMDB

# BOW
python -m src.experiments.TestSingleSplit --class1=1 --ngram_e=3 --preprocessing_path='datasets/preprocess/imdb_not_preprocessed.csv' --classifierType='bernoulli' --name='IMDBBernoulliBOW(1,3)105' --seed=105 --show=False
echo
python -m src.experiments.TestSingleSplit --class1=1 --ngram_e=3 --preprocessing_path='datasets/preprocess/imdb_not_preprocessed.csv' --classifierType='multinomial' --multinomial=True --name='IMDBMultinomialBOW(1,3)105' --seed=105 --show=False
echo
# Tfidf
python -m src.experiments.TestSingleSplit --class1=1 --ngram_e=3 --preprocessing_path='datasets/preprocess/imdb_not_preprocessed.csv' --classifierType='multinomial' --multinomial=True --tfidf=True --name='IMDBMultinomialTfidfBOW(1,3)105' --seed=105 --show=False
echo

# Fasttext
python -m src.experiments.TestSingleSplit --class1=1 --bow=False --ngram_e=3 --preprocessing_path='datasets/preprocess/imdb_not_preprocessed.csv' --classifierType='multinomial' --multinomial=True --emb_import_path='datasets/fasttext/imdb_embedding.ft' --name='IMDBMultinomialFastText(1,3)105' --seed=105 --show=False
echo
python -m src.experiments.TestSingleSplit --class1=1 --bow=False --ngram_e=3 --preprocessing_path='datasets/preprocess/imdb_not_preprocessed.csv' --classifierType='multinomial' --multinomial=True --emb_import_path='datasets/fasttext/imdb_embedding.ft' --tfidf=True --name='IMDBMultinomialTfidfFastText(1,3)105' --seed=105 --show=False
echo
python -m src.experiments.TestSingleSplit --class1=1 --bow=False --ngram_e=3 --preprocessing_path='datasets/preprocess/imdb_not_preprocessed.csv' --classifierType='gaussian' --emb_import_path='datasets/fasttext/imdb_embedding.ft' --name='IMDBGaussianFastText(1,3)105' --seed=105 --show=False
echo
python -m src.experiments.TestSingleSplit --class1=1 --bow=False --ngram_e=3 --preprocessing_path='datasets/preprocess/imdb_not_preprocessed.csv' --classifierType='categorical' --tfidf=True --emb_import_path='datasets/fasttext/imdb_embedding.ft' --name='IMDBCategoricalFastText(1,3)105' --seed=105 --show=False
echo

# Imdb v Twitter - Twitter v Imdb
python -m src.experiments.ImdbTwitterSingleSplit --ngram_e=3 --path_imdb='datasets/preprocess/imdb_not_preprocessed.csv' --path_tweet='datasets/preprocess/twitter_not_preprocessed.csv' --name='BernoulliBOW(1,3)105' --seed=105 --show=False
echo

# Reddit

# BOW
python -m src.experiments.TestSingleSplit --class1=1 --ngram_e=3 --preprocessing_path='datasets/preprocess/reddit_not_preprocessed.csv' --classifierType='bernoulli' --name='RedditBernoulliBOW(1,3)105' --seed=105 --show=False
echo
python -m src.experiments.TestSingleSplit --class1=1 --ngram_e=3 --preprocessing_path='datasets/preprocess/reddit_not_preprocessed.csv' --classifierType='multinomial' --multinomial=True --name='RedditMultinomialBOW(1,3)105' --seed=105 --show=False
echo
# Tfidf
python -m src.experiments.TestSingleSplit --class1=1 --ngram_e=3 --preprocessing_path='datasets/preprocess/reddit_not_preprocessed.csv' --classifierType='multinomial' --multinomial=True --tfidf=True --name='RedditMultinomialTfidfBOW(1,3)105' --seed=105 --show=False
echo

# Fasttext
python -m src.experiments.TestSingleSplit --class1=1 --bow=False --ngram_e=3 --preprocessing_path='datasets/preprocess/reddit_not_preprocessed.csv' --classifierType='multinomial' --multinomial=True --emb_import_path='datasets/fasttext/reddit_embedding.ft' --name='RedditMultinomialFastText(1,3)105' --seed=105 --show=False
echo
python -m src.experiments.TestSingleSplit --class1=1 --bow=False --ngram_e=3 --preprocessing_path='datasets/preprocess/reddit_not_preprocessed.csv' --classifierType='multinomial' --multinomial=True --emb_import_path='datasets/fasttext/reddit_embedding.ft' --tfidf=True --name='RedditMultinomialTfidfFastText(1,3)105' --seed=105 --show=False
echo
python -m src.experiments.TestSingleSplit --class1=1 --bow=False --ngram_e=3 --preprocessing_path='datasets/preprocess/reddit_not_preprocessed.csv' --classifierType='gaussian' --emb_import_path='datasets/fasttext/reddit_embedding.ft' --name='RedditGaussianFastText(1,3)105' --seed=105 --show=False
echo
python -m src.experiments.TestSingleSplit --class1=1 --bow=False --ngram_e=3 --preprocessing_path='datasets/preprocess/reddit_not_preprocessed.csv' --classifierType='categorical' --tfidf=True --emb_import_path='datasets/fasttext/reddit_embedding.ft' --name='RedditCategoricalFastText(1,3)105' --seed=105 --show=False
echo
