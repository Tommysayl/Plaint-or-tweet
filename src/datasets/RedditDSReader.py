from src.datasets.DatasetReader import DatasetReader
import os
import pandas as pd
import re
import numpy as np


class RedditDSReader(DatasetReader):

    @property
    def default_path(self) -> str:
        return '../../datasets/reddit.csv'

    def read_from_file(self, p=None, remove_stopwords=True, remove_links=True, correct_typos=True) -> None:
        super().read_from_file(p)

        df = pd.read_csv(self.get_path(p), encoding="ISO-8859-1", names=["id", "timestamp", "team", "subreddit", "sentiment", "text"])

        for i in range(len(df["text"])):
            if type(df["text"][i]) != str:
                df["text"][i] = str(df["text"][i])

        comments = self.preprocess(df["text"][1:], remove_links, correct_typos)

        for i in range(1, len(df["sentiment"])):
            df["sentiment"][i] = 4 if float(df["sentiment"][i]) < 0 else 0

        self.build_from(comments, df["sentiment"][1:], remove_stopwords)
