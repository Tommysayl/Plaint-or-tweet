from abc import ABC, abstractmethod
import os
from src.utils import disable_exception_traceback, reduce_lengthening
import spacy
from collections import namedtuple
from typing import Generator, List
import re
from urlextract import URLExtract
from string import punctuation
from src.datasets import TwitterDSReader

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
    def default_path(self) -> str:
        ''' A DatasetReader must provide the relative path to its dataset. '''
        pass

    @abstractmethod
    def read_from_file(self, p=None, remove_stopwords=True, remove_links=True, correct_typos=True) -> None:
        ''' Implement here the logic for reading and parsing the dataset.  '''
        if not os.path.exists(self.get_path(p)):
            with disable_exception_traceback():
                raise IOError(self.WRONG_PATH_ERR.format(self.get_path(p)))

    def build_from(self, docs, labels, remove_stopwords) -> None:
        ''' Builds the dataset as a collection of [DatasetInstance]s from provided data. '''
        if remove_stopwords:
            self._ds = (DatasetIstance(self.clear_stopwords(doc), label) for doc, label in zip(docs, labels))
        else:
            self._ds = (DatasetIstance([token for token in doc if not token.text.isspace()], label) for doc, label in zip(docs, labels))

    def clear_stopwords(self, doc) -> List[str]:
        ''' Returns a list of non-stopword tokens. '''
        return [token for token in doc if token.text not in self.nlp.Defaults.stop_words and not token.text.isspace()]

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
            sentencizer = self._nlp.create_pipe("sentencizer")
            self._nlp.add_pipe(sentencizer)
            self.nlp.Defaults.stop_words |= set(punctuation)    # not sure if it's worth
        return self._nlp

    def docs(self) -> DatasetIstance:
        ''' Returns an iterator of [DatasetIstance]s. '''
        if self._ds is None:
            self.read_from_file()
        return (istance for istance in self._ds if len(istance.tokens))

    def get_path(self, path) -> str:
        ''' Returns the provided dataset [path] if it exists, otherwise will return the [default_path] of the required dataset.  '''
        dir = os.path.dirname(os.path.abspath(__file__))
        return path if path is not None else os.path.join(dir, self.default_path)

    def preprocess(self, raw_text, remove_links, correct_typos) -> Generator[spacy.tokens.doc.Doc, None, None]:
        ''' Run base nlp pipeline over [raw_text], including tokenization, lemmatization and utilities to correct user typos. '''
        batch_text = self._reduce_text_noise(raw_text, remove_links, correct_typos)
        return self.nlp.pipe(batch_text)

    def _reduce_text_noise(self, text, remove_links, correct_typos) -> Generator[str, None, None]:
        ''' Just a vectorized function that applies _reduce_span_noise() to every span of the provided text. '''
        url_extractor = URLExtract()
        return (self._reduce_span_noise(span, url_extractor, remove_links, correct_typos) for span in text)

    def _reduce_span_noise(self, span, extractor, remove_links, correct_typos) -> str:
        if isinstance(self, TwitterDSReader.TwitterDSReader):
            span = self._extract_hashtags(span)
        ''' Apply some grammar corrections to fix user typos or remove URLs from text. '''
        if remove_links:
            span = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', span)
            #for url in extractor.find_urls(span):
                #span = span.replace(url, "")
        if correct_typos:
            span = reduce_lengthening(span)
        return re.sub(' +', ' ', span)
