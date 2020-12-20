from src.datasets.DatasetReader import DatasetReader
import os
import pandas as pd
import re

class ImdbDSReader(DatasetReader):

    @property
    def default_path(self) -> str:
        return "../../datasets/imdb.csv"

    def read_from_file(self, p=None, remove_stopwords=True, remove_links=True, correct_typos=True) -> None:
        super().read_from_file(p)

        df = pd.read_csv(self.get_path(p), encoding="ISO-8859-1", names=["text", "label"])
        reviews = self.preprocess(df["text"], remove_links, correct_typos)

        self.build_from(reviews, df["label"], remove_stopwords)