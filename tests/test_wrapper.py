import numpy as np

from links_cluster import EvictingCacheWrapper

class TestEvictingCache:
    """Tests for EvictingCacheWrapper class."""
    def setup_method(self):
        self.size = 10
        self.class_n = 5
        self.cache = EvictingCacheWrapper(0.1, 0.05, 1.0, True, self.size, False)
        self.cache_dense = EvictingCacheWrapper(
            cluster_similarity_threshold=0.2, subcluster_similarity_threshold=0.10, pair_similarity_maximum=1.0,
            query_whole_cluster=True, max_size=1000, evict_unsafe=True)

    def test_push_and_eviction(self):
        for i in range(self.size*10):
            assert(len(self.cache.stored_vectors) == min(self.size, i))
            self.cache.push(new_key=i, new_vector=np.eye(self.class_n)[i % self.class_n], top_n=0)
        test = self.cache.push(new_key=100, new_vector=np.eye(self.class_n)[0], top_n=10)
        assert(all([x[0] == 1 for x in test]))
        # assert(self.size//self.class_n <= len(test) <= self.size//self.class_n + 1)

    def test_intense_eviction(self):
        # makes 30 clusters for 100 times, and then use only 15 clusters
        data = ((i, np.eye(30)[i % 30 if i < 100 else i % 15]) for i in range(1500))
        for k, x in data:
            _ = self.cache_dense.push(k, x, top_n=10)

    def test_intense_merge(self):
        # TODO : subcluster index error when subclusters are merging
        pass