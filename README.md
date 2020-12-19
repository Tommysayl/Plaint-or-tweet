# Plaint-or-tweet
Final project for the Fundamentals of Data Science course.

<img src="logo.png" width="256px">

# Install dependencies
- Download and install [Miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)
- Run ```conda env create -f environment.yml```
- Run ```conda activate plaint_or_tweet```
- That's it!

# Update dependencies
- Add a new dependency in environment.yml
- Run ```conda env update -f environment.yml --prune```

# Extract the datasets
To be able to test our models you need to extract at least one of the available datasets:
- [Sentiment140](https://www.kaggle.com/kazanova/sentiment140): ```unzip -d datasets/ sentiment140.zip```
- [Twitter60k](https://github.com/Sanjay-Suthraye/twitter_sentiment_analysis): ```unzip -d datasets/ datasets/twitter60k.zip```
- [IMDb](https://www.kaggle.com/kazanova/sentiment140): ```unzip -d datasets/ datasets/imdb.zip```

# Authors
- [Tommaso Battistini](https://github.com/Frisayl)
- [Edoardo De Matteis](https://github.com/edodema)
- [Leonardo Emili](https://github.com/LeonardoEmili)
- [Mirko Giacchini](https://github.com/Mirko222)
- [Alessio Luciani](https://github.com/AlessioLuciani)
