import fire
import numpy as np
import pandas as pd
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

def load_preprocessing(path):
    df = pd.read_csv(path)
    corpus = df.iloc[:,2].values
    y = df['Label'].values
    return corpus, y

def main(dataset_path = 'datasets/preprocess/twitter_preprocessed.csv', maxFontSize=125, maxWords=100, width=1400, height=700, backgroundColor="white", labelToPrint=4):

    corpus, label = load_preprocessing(dataset_path)
    text = [a for i,a in enumerate(corpus) if label[i] == labelToPrint]
    text = ' '.join(text)

    wordcloud = WordCloud(max_font_size=maxFontSize, width=width, height=height, max_words=maxWords, background_color=backgroundColor).generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

if __name__ == '__main__':
    fire.Fire(main)