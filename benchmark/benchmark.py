from links_clustering import EvictingCacheWrapper
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd


def encode(idxs):
    tags = np.zeros(len(tags_info))
    tags[idxs] = 1
    return tags


#################
## Time benchmark
cache = EvictingCacheWrapper(cluster_similarity_threshold=0.1, subcluster_similarity_threshold=0.05,
                             pair_similarity_maximum=1.0, query_whole_cluster=True, max_size=1000)
i, times = 0, []
for k, x in a[1].items():
    _ = cache.push(k, encode(x["tags_"]), top_n=10)
    times.append(time.time())
    i += 1
    if i % 1000 == 0:
        print(i)

cache = EvictingCacheWrapper(cluster_similarity_threshold=0.1, subcluster_similarity_threshold=0.05,
                             pair_similarity_maximum=1.0, query_whole_cluster=True, max_size=1000)
i, times_shuffle = 0, []
aa = [(k, v) for k, v in a[1].items()]
np.random.shuffle(aa)
for k, x in aa:
    _ = cache.push(k, encode(x["tags_"]), top_n=10)
    times_shuffle.append(time.time())
    i += 1
    if i % 1000 == 0:
        print(i)

pd.DataFrame({"noshuffle": times, "shuffle": times_shuffle}).diff().rolling(100).mean().plot()
plt.title("Time per query")

#######################
## Similarity benchmark
l1_distance = []
for x1, x2 in zip(list(a[1].values())[1:], list(a[1].values())[:-1]):
    l1_distance.append(np.sum(np.abs(encode(x1["tags_"]) - encode(x2["tags_"]))))
avg_l1_distance = np.mean(l1_distance)

i = 0
l1_distance = []
for k, x in a[1].items():
    tag = encode(x["tags_"])
    result = cache.push(k, tag, top_n=10)
    l1_distance.append(np.mean([np.sum(np.abs(x - tag)) for x in result]))
    i += 1
    if i % 1000 == 0:
        print(i)
pd.Series(l1_distance).rolling(100).mean().plot()
plt.axhline(avg_l1_distance, color="orange")
plt.ylim(0, 60)
plt.title("L1 distance vs. average")
