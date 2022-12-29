from links_clustering import EvictingCacheWrapper
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

def encode(idxs):
    tags = np.zeros(len(tags_info))
    tags[idxs] = 1
    return tags

def cluster_stat(cache):
    sc_n = [sc.n_vectors for c in cache.clusters for sc in c]
    return {
        "num_cluster":len(cache.clusters),
        "avg_subcluster_size": np.mean(sc_n),
        "std_subcluster_size": np.std(sc_n)
    }

def make_cache(evict):
    cache = EvictingCacheWrapper(
        cluster_similarity_threshold=0.5,subcluster_similarity_threshold=0.2,pair_similarity_maximum=1.0,query_whole_cluster=True,max_size=1000,
        remove_old_unsafe=evict
    )
    return cache

def loop(data, cache, cb, debug=False):
    i = 0
    output = []
    for k,x in data:
        try:
            result = cache.push(k, x, top_n=10)
            if debug :
                pass
        except:
            raise ValueError(f"k:{k}, x:{x.tolist()}")
        output.append(cb(cache, (result,x)))
        i += 1
        if i%1000 == 0 :
            print(i)
    return output


#################
## Time benchmark
time_bench_evict = loop(((k,encode(v["tags_"])) for k,v in datagen_block[1].items()), make_cache(evict=True), lambda _, _ : {"time_evict":time.time()}, debug=False)
time_bench_noevict = loop(((k,encode(v["tags_"])) for k,v in datagen_block[1].items()), make_cache(evict=False), lambda _, _ : {"time_noevict":time.time()}, debug=False)
datagen_shuffle = [(k,v) for k,v in datagen_block[1].items()]
np.random.shuffle(datagen_shuffle)
time_bench_shuffle = loop(((k,encode(v["tags_"])) for k,v in datagen_shuffle), make_cache(evict=True), lambda _, _ : {"time_evict_shuffle":time.time()}, debug=False)
df_time = pd.concat([pd.DataFrame(time_bench_evict),pd.DataFrame(time_bench_shuffle),pd.DataFrame(time_bench_noevict)], axis=1).diff()
df_time.rolling(100).mean().plot(title="Time per query")

#######################
## Cluster benchmark
cluster_bench_evict = loop(((k,encode(v["tags_"])) for k,v in datagen_block[1].items()), make_cache(evict=True), lambda cache, _ : cluster_stat(cache), debug=False)
cluster_bench_noevict = loop(((k,encode(v["tags_"])) for k,v in datagen_block[1].items()), make_cache(evict=False), lambda cache, _ : cluster_stat(cache), debug=False)
df_cluster = pd.concat([pd.DataFrame(cluster_bench_evict).add_prefix("evict_"),pd.DataFrame(cluster_bench_noevict).add_prefix("noevict_")], axis=1)
df_cluster.rolling(10).mean().plot(title="Cluster per query")

#######################
## Similarity benchmark
from scipy.spatial.distance import cosine
cosine_distance = []
for x1, x2 in zip(list(datagen_block[1].values())[1:], list(datagen_block[1].values())[:-1]):
    cosine_distance.append(cosine(encode(x1["tags_"]),encode(x2["tags_"])))
avg_cosine_distance = np.mean(cosine_distance)

similarity_bench_evict = loop(((k,encode(v["tags_"])) for k,v in datagen_block[1].items()), make_cache(evict=True), lambda _, res : np.mean([cosine(x,res[1]) for x in res[0]]), debug=False)
similarity_bench_noevict = loop(((k,encode(v["tags_"])) for k,v in datagen_block[1].items()), make_cache(evict=False), lambda _, res : np.mean([cosine(x,res[1]) for x in res[0]]), debug=False)
df_similarity = pd.DataFrame({"cosine_distance_evict": similarity_bench_evict, "cosine_distance_noevict":similarity_bench_noevict, "cosine_distance_avg":avg_cosine_distance})
df_similarity.rolling(100).mean().plot(title="Similarity per query", ylim=(0,None))