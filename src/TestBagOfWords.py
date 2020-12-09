from BernoulliNaiveBayes import BernoulliNaiveBayes
from DatasetBagOfWords import DatasetBagOfWords
from TwitterDSReader import TwitterDSReader

bow = DatasetBagOfWords()
bow.bag_of_words(TwitterDSReader())

bow.save('bow.csv')

#print('Write a tweet:')
#tweet = input()

#print(bow.embed(tweet))

#model = BernoulliNaiveBayes()
#model.train(bow.X, bow.y)