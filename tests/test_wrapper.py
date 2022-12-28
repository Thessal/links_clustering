import numpy as np

from links_cluster import EvictingCacheWrapper

class TestEvictingCache:
    """Tests for EvictingCacheWrapper class."""
    def setup_method(self):
        self.size = 10
        self.class_n = 5
        self.cache = EvictingCacheWrapper(0.1, 0.05, 1.0, True, self.size)

    def test_push_and_eviction(self):
        for i in range(self.size*10):
            assert(len(self.cache.stored_vectors) == min(self.size, i))
            self.cache.push(new_key=i, new_vector=np.eye(self.class_n)[i % self.class_n], top_n=0)
        test = self.cache.push(new_key=100, new_vector=np.eye(self.class_n)[0], top_n=10)
        assert(all([x[0] == 1 for x in test]))
        assert(self.size//self.class_n <= len(test) <= self.size//self.class_n + 1)