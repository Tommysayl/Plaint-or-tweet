from abc import ABC, abstractmethod
import os
from utils import disable_exception_traceback
import spacy
from collections import namedtuple
from typing import Generator
import re
from urlextract import URLExtract

DatasetIstance = namedtuple("DatasetIstance", "tokens label")

class Singleton(object):
    _instance = None

    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance


class DatasetReader(Singleton, ABC):
    WRONG_PATH_ERR = "No dataset found at {}.\nPlease provide a path to an available dataset or use the default one."

    _nlp = None
    _ds = None

    @property
    @abstractmethod
    def default_path(self):
        ''' A DatasetReader must provide the relative path to its dataset. '''
        pass

    @abstractmethod
    def read_from_file(self, p=None) -> None:
        ''' Implement here the logic for reading and parsing the dataset.  '''
        pass

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
            sentencizer = self._nlp.create_pipe("sentencizer")
            self._nlp.add_pipe(sentencizer)
        return self._nlp

    def validate_dataset(self, p=None) -> None:
        if not os.path.exists(self.get_path(p)):
            with disable_exception_traceback():
                raise IOError(self.WRONG_PATH_ERR.format(p))

    def docs(self) -> DatasetIstance:
        if self._ds is None:
            self.read_from_file()
        return self._ds

    def get_path(self, path) -> str:
        dir = os.path.dirname(os.path.abspath(__file__))
        return path if path is not None else os.path.join(dir, self.default_path)

    def reduce_text_noise(self, text, remove_stopwords, remove_links, correct_typos) -> Generator[str, None, None]:
        url_extractor = URLExtract()
        return (self.reduce_span_noise(span, url_extractor, remove_stopwords, remove_links, correct_typos) for span in text)

    def reduce_span_noise(self, span, extractor, remove_stopwords, remove_links, correct_typos) -> str:
        if remove_stopwords:
            # TODO: implement this
            pass
        if remove_links:
            for url in extractor.find_urls(span):
                span = span.replace(url, "")
        if correct_typos:
            # TODO: implement this
            pass
        return re.sub(' +', ' ', span)
