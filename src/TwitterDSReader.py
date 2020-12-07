from DatasetReader import DatasetReader
import os
import pandas as pd
import sys
import spacy
from DatasetReader import DatasetIstance


class TwitterDSReader(DatasetReader):

    @property
    def default_path(self):
        return "../datasets/training.1600000.processed.noemoticon.csv"

    def read_from_file(self, p=None, remove_stopwords=True, remove_links=True, correct_typos=True) -> None:
        self.validate_dataset(p)
        df = pd.read_csv(self.get_path(p), encoding="ISO-8859-1", header=None)
        df.columns = ["label", "time", "date", "query", "username", "text"]
        tweets, labels = df["text"], df["label"]

        tweets = self.reduce_text_noise(tweets, remove_stopwords, remove_links, correct_typos)

        tweets = self.nlp.pipe(tweets)
        self._ds = (DatasetIstance(tweet.sents, label) for tweet, label in zip(tweets, labels))