import numpy as np

from cache_wrapper import EvictingCacheWrapper

class TestEvictingCache:
    """Tests for EvictingCacheWrapper class."""
    def setup_method(self):
        self.cache = EvictingCacheWrapper()

