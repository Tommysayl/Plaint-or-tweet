from DatasetBagOfWords import DatasetBagOfWords
import math
from TwitterDSReader import TwitterDSReader


bow = DatasetBagOfWords()
bow.load_preprocessing('bow_preprocess.csv')
bow.train(0.1)

#corpus = bow.load_preprocessing('bow_preprocess.csv')

#bow.train(0.001)

#print(len(bow.X))
#print(len(bow.y))

#bow.save_bag_of_words('bow.csv')

#print(bow.X)
#print(bow.vocabulary)