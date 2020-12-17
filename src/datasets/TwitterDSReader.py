from src.datasets.DatasetReader import DatasetReader
import os
import pandas as pd
import re


class TwitterDSReader(DatasetReader):

    @property
    def default_path(self) -> str:
        #return "../../datasets/training.1600000.processed.noemoticon.csv"
        return "../../datasets/twitter.csv"

    def read_from_file(self, p=None, remove_stopwords=True, remove_links=True, correct_typos=True) -> None:
        super().read_from_file(p)

        df = pd.read_csv(self.get_path(p), encoding="ISO-8859-1", names=["label", "id", "date", "query", "username", "text"])
        tweets = self.preprocess(df["text"], remove_links, correct_typos)

        self.build_from(tweets, df["label"], remove_stopwords)

    def _extract_hashtags(self, tweet) -> str:
        ''' Source: https://www.kaggle.com/mistryjimit26/twitter-sentiment-analysis-basic '''
        return re.sub(r'#([^\s]+)', r'\1', tweet)