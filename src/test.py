import pickle

with open("export/bernoulli_bow.potmodel", "rb") as fi:
    print(pickle.load(fi))