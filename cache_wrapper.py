import numpy as np
from .links_cluster import LinksCluster, Subcluster
from scipy.spatial.distance import cosine
import itertools
from collections import defaultdict


class Item:
    def __init__(self, key, vector, cl_idx, sc_idx, vec_idx, next_item=None):
        self.key = key
        self.vector = vector
        self.cl_idx = cl_idx
        self.sc_idx = sc_idx
        self.vec_idx = vec_idx
        self.next_item = next_item


class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.n = 0
        self.clusters = defaultdict(lambda: defaultdict(lambda: {}))

    def __len__(self):
        return self.n

    def push(self, key, vector, cl_idx, sc_idx, vec_idx):
        if self.head:
            self.tail.next_item = Item(key, vector, cl_idx, sc_idx, vec_idx)
            self.tail = self.tail.next_item
        else:
            self.head = Item(key, vector, cl_idx, sc_idx, vec_idx)
            self.tail = self.head
        self.clusters[cl_idx][sc_idx][vec_idx] = self.tail
        self.n += 1

    def pop(self):
        head = self.head
        self.head = self.head.next_item
        self.n -= 1
        self.clusters[head.cl_idx][head.sc_idx].pop(head.vec_idx)
        return head

    def modify(self, cl_1, sc_1, vec_1, cl_2, sc_2, vec_2):
        if vec_1 is None:
            if (sc_2 in self.clusters[cl_1]) and len(self.clusters[cl_2][sc_2]) > 0:
                raise ValueError(
                    f"modifying {cl_1} {sc_1} to {cl_2} {sc_2}, but subcluster {sc_2} exists and contains {len(self.clusters[cl_2][sc_2])} items")
            sc = self.clusters[cl_1].pop(sc_1)
            self.clusters[cl_2][sc_2] = sc
            for item in sc.values():
                item.cl_idx, item.sc_idx = cl_2, sc_2
        else:
            item = self.clusters[cl_1][sc_1].pop(vec_1)
            self.clusters[cl_2][sc_2][vec_2] = item
            item.cl_idx, item.sc_idx, item.vec_idx = cl_2, sc_2, vec_2

    def __iter__(self):
        return LinkedListIterator(self.head)


class LinkedListIterator:
    def __init__(self, head):
        self.current = head

    def __iter__(self):
        return self

    def __next__(self):
        if not self.current:
            raise StopIteration
        else:
            item = self.current
            self.current = self.current.next_item
            return item


class EvictingCacheWrapper(LinksCluster):
    def __init__(self, cluster_similarity_threshold: float, subcluster_similarity_threshold: float,
                 pair_similarity_maximum: float, query_whole_cluster: bool, max_size: int, evict_unsafe: bool):
        super().__init__(
            cluster_similarity_threshold, subcluster_similarity_threshold, pair_similarity_maximum,
            store_vectors=True if evict_unsafe else False, evict_unsafe=evict_unsafe
        )
        self.stored_vectors = LinkedList()
        # TODO : optimize, use ndarray rather than list
        self.query_whole_cluster = query_whole_cluster
        self.max_size = max_size

    def query_vector(self):
        item = self.stored_vectors.tail
        for sc in (
                sorted(self.clusters[item.cl_idx], key=lambda sc_: cosine(item.vector, sc_.centroid))
                if self.query_whole_cluster
                else self.clusters[item.cl_idx][item.sc_idx]
        ):
            for vec in sorted(sc.input_vectors, key=lambda vec_: cosine(item.vector, vec_)):
                yield vec

    # def dump(self):
    #     print("==")
    #     for x in self.stored_vectors:
    #         print(x.key, x.cl_idx, x.sc_idx, x.vec_idx, self.clusters[x.cl_idx][x.sc_idx].connected_subclusters)
    #     print("==")

    def push(self, new_key: int, new_vector: np.ndarray, top_n=10) -> list:
        super().predict_(new_key, new_vector, wrapper_stored_vectors=self.stored_vectors)
        if len(self.stored_vectors) > self.max_size:
            self.pop()
        return list(itertools.islice(self.query_vector(), top_n))

    def pop(self):
        super().remove_(wrapper_stored_vectors=self.stored_vectors)
