
import numpy as np
from MultinomialNaiveBayes import MultinomialNaiveBayes

def test_binary():

    X = np.array([
        [1, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 1, 1, 1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        ])

    y = np.array([1, 
                1, 
                0, 
                1, 
                0,
                1,
                1,
                0])


    model = MultinomialNaiveBayes()
    model.train(X, y)
    print(model.predict_class([0, 0, 1, 1]))

def test_multinomial():

    X = np.array([
        [2, 3, 1, 0],
        [0, 0, 3, 3],
        [1, 2, 0, 0],
        [3, 3, 0, 1],
        [0, 1, 2, 2],
        [0, 0, 1, 1],
        [0, 2, 3, 3],
        [1, 3, 1, 0],
        ])

    y = np.array([0, 
                1, 
                0, 
                0, 
                1,
                1,
                1,
                0])


    model = MultinomialNaiveBayes()
    model.train(X, y)
    print(model.predict_class([0, 0, 0, 3]))
    print(model.predict_class([1, 3, 1, 0]))
    print(model.predict_class([2, 2, 0, 0]))
    print(model.predict_class([1, 2, 3, 2]))
    print(model.predict_class([1, 0, 2, 3]))


if __name__ == "__main__":
    test_multinomial()