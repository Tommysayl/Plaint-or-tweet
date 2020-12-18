from src.datasets.Embedder import Embedder
import time

start_time = time.time()
csv = "twitter_preprocess.csv"
vector_size = 100
window_size = 5
sg = 0
min_count = 5
workers = 1

embedder = Embedder()
print("creating FastText model... ")
ft_model = embedder.embed_ft(csv, vector_size, window_size, min_count, workers, sg)
print("Done")
print("Time needed:", (time.time() - start_time))
print("Creating Word2Vec model...")
#w2v_model = embedder.embed_w2v(csv, vector_size, window_size, min_count, workers, sg)
print("Done")

print("Time needed:", (time.time() - start_time))
print("Comparing models: science:")
print("FastText:")
print(ft_model.wv.most_similar("science"))
print("Word2Vec:")
#print(w2v_model.wv.most_similar("science"))
print("Second comparison: homework")
print("FastText:")
print(ft_model.wv.most_similar("homework"))
print("Word2Vec:")
#print(w2v_model.wv.most_similar("homework"))
print("Time needed:", (time.time() - start_time))
print("Discretizing embedded vectors:")
ciao = embedder.discretize_vectors(ft_model)
print("Testing Embedding discretization:")
print("Original vector for the word Science:")
print(ft_model.wv["science"])
print("Discretized Vector:")
print(ciao["science"])
print("total time for test:",(time.time() - start_time))


