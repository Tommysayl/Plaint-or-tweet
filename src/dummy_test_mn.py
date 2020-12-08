import numpy as np
import collections, re
import pandas as pd
import random
from MultinomialNaiveBayes import MultinomialNaiveBayes


def test_binary():

    X = np.array(
        [
            [1, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 1, 1, 1],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
        ]
    )

    y = np.array([1, 1, 0, 1, 0, 1, 1, 0])

    model = MultinomialNaiveBayes()
    model.train(X, y)
    print(model.predict_class([0, 0, 1, 1]))


def test_multinomial():

    X = np.array(
        [
            [2, 3, 1, 0],
            [0, 0, 3, 3],
            [1, 2, 0, 0],
            [3, 3, 0, 1],
            [0, 1, 2, 2],
            [0, 0, 1, 1],
            [0, 2, 3, 3],
            [1, 3, 1, 0],
        ]
    )

    y = np.array([0, 1, 0, 0, 1, 1, 1, 0])

    model = MultinomialNaiveBayes()
    model.train(X, y)
    print(model.predict_class([0, 0, 0, 3]))
    print(model.predict_class([1, 3, 1, 0]))
    print(model.predict_class([2, 2, 0, 0]))
    print(model.predict_class([1, 2, 3, 2]))
    print(model.predict_class([1, 0, 2, 3]))


def extreme_test():
    n_inst = 10000  # Like 10k tweets
    n_feat = 10000  # Like 10k words of English vocabulary
    X = np.array([[random.randint(0, 5) for j in range(n_feat)] for i in range(n_inst)])
    y = np.array([random.randint(0, 1) for i in range(n_inst)])
    model = MultinomialNaiveBayes()
    model.train(X, y)

    for i in range(10):
        print(model.predict_class([random.randint(0, 5) for j in range(n_feat)]))


def imbd_test():
    # using https://www.kaggle.com/columbine/imdb-dataset-sentiment-analysis-in-csv-format
    tr = pd.read_csv("../datasets/imbd/Train.csv")

    # using https://github.com/dwyl/english-words/blob/master/words_alpha.zip
    words = pd.read_csv("../datasets/words_alpha.txt", header=None, encoding="utf-8")

    dic = {}
    i = 0
    for word in words[0]:
        dic[word] = i
        i += 1

    n_feat = len(dic)

    n_inst = min(1000, len(tr["text"]))
    X = np.zeros((n_inst, n_feat))
    y = np.zeros(n_inst)
    for i in range(n_inst):
        y[i] = tr["label"][i]
        split = tr["text"][i].split(" ")
        indeces = []
        print(i)
        for word in split:
            if word in dic:
                indeces.append(dic[word])
        indeces.sort()
        z = 0  # running index
        for j in range(n_feat):
            if z < len(indeces) and indeces[z] == j:
                X[i][j] = 1
                z += 1
            else:
                X[i][j] = 0

    model = MultinomialNaiveBayes()
    model.train(X.astype(int), y.astype(int))

    # Testing
    test = pd.read_csv("../datasets/imbd/Test.csv")

    n_inst_test = min(100, len(test["text"]))

    X_test = np.zeros((n_inst_test, n_feat))
    y_test = np.zeros((n_inst_test))

    for i in range(n_inst_test):
        y_test[i] = test["label"][i]
        split = test["text"][i].split(" ")
        indeces = []
        print(i)
        for word in split:
            if word in dic:
                indeces.append(dic[word])
        indeces.sort()
        z = 0  # running index
        for j in range(n_feat):
            if z < len(indeces) and indeces[z] == j:
                X_test[i][j] = 1
                z += 1
            else:
                X_test[i][j] = 0

    # predicting!
    n_correct = 0
    for i in range(n_inst_test):
        pred = model.predict_class(X[i].astype(int))
        print(y_test[i], " -> ", pred)
        if y_test[i] == pred:
            n_correct += 1

    print("Accuracy: ", n_correct / n_inst_test)


if __name__ == "__main__":
    # test_multinomial()
    # extreme_test()
    imbd_test()