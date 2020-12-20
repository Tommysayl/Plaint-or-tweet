import fire
import pickle
from src.models.BernoulliNaiveBayes import BernoulliNaiveBayes

def main(hashtag="election", model_path="export/bernoulli_bow.potmodel"):

    # Calling twitter API to retrieve tweets by hashtag
    tweets = [] # TODO: API call

    #TODO: preprocessing

    # Loading model
    model = None
    with open(model_path, "rb") as fi:
        model = BernoulliNaiveBayes(from_dict = pickle.load(fi)) 


    #TODO: make prediction
    print("Prediction")


if __name__ == "__main__":
    fire.Fire(main)