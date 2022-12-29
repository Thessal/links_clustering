import numpy as np
from .links_cluster import LinksCluster, Subcluster
from scipy.spatial.distance import cosine
import itertools


class EvictingCacheWrapper(LinksCluster):
    def __init__(self, cluster_similarity_threshold: float, subcluster_similarity_threshold: float,
                 pair_similarity_maximum: float, query_whole_cluster: bool, max_size: int):
        super().__init__(
            cluster_similarity_threshold, subcluster_similarity_threshold, pair_similarity_maximum,
            store_vectors=True,
        )
        self.stored_vectors = []  # [new_key, new_vector, cluster_idx, subcluster_idx, vector_idx]
        # TODO : optimize, use ndarray rather than list
        self.query_whole_cluster = query_whole_cluster
        self.max_size = max_size

    def query_vector(self, stored_vectors_idx=-1):
        item = self.stored_vectors[stored_vectors_idx]
        _, target_vector, cluster_idx, subcluster_idx, _ = item
        for sc in (
                sorted(self.clusters[cluster_idx], key=lambda sc: cosine(target_vector, sc.centroid))
                if self.query_whole_cluster
                else self.clusters[cluster_idx][subcluster_idx]
        ):
            for vec in sorted(sc.input_vectors, key=lambda vec: cosine(target_vector, vec)):
                yield vec

    def merge_subclusters(self, cl_idx, sc_idx1, sc_idx2):
        vec_offset = self.clusters[cl_idx][sc_idx1].n_vectors
        super().merge_subclusters(cl_idx, sc_idx1, sc_idx2)
        for i, item in enumerate(self.stored_vectors):
            _, _, cluster_idx, subcluster_idx, _ = item
            if cluster_idx == cl_idx:
                if subcluster_idx == sc_idx2:
                    self.stored_vectors[i][3] = sc_idx1
                    self.stored_vectors[i][4] += vec_offset
                if subcluster_idx > sc_idx2:
                    self.stored_vectors[i][3] -= 1

    def push(self, new_key: int, new_vector: np.ndarray, top_n=10) -> list:
        cluster_idx, subcluster_idx, vector_idx = super().predict_subcluster(new_vector)
        if len(self.stored_vectors) >= self.max_size:
            self.pop()
        self.stored_vectors.append([new_key, new_vector, cluster_idx, subcluster_idx, vector_idx])
        return list(itertools.islice(self.query_vector(-1), top_n))

    def check_integrity(self):
        # Check hollow subcluster
        assert(all([(sc.centroid is not None) for c in self.clusters for sc in c]))
        assert(all([
            (csc.centroid is not None) for c in self.clusters for sc in c for csc in sc.connected_subclusters
        ]))

        # Check indexer
        for _, target_vector, cluster_idx, subcluster_idx, vector_idx in self.stored_vectors:
            assert(
                    len(self.clusters[cluster_idx][subcluster_idx].input_vectors) ==
                    self.clusters[cluster_idx][subcluster_idx].n_vectors
            )
        for k, target_vector, cluster_idx, subcluster_idx, vector_idx in self.stored_vectors:
            if not (self.clusters[cluster_idx][subcluster_idx].n_vectors > vector_idx):
                for x in self.stored_vectors:
                    print(x)
                print()
                print(k, target_vector, cluster_idx, subcluster_idx, vector_idx)
                print()
                for i,cl in enumerate(self.clusters):
                    for j,sc in enumerate(cl):
                        print(i, j, sc.input_vectors, sc.n_vectors)
                raise Exception()

    def pop(self, stored_vectors_idx=0):
        item = self.stored_vectors.pop(stored_vectors_idx)
        _, target_vector, cluster_idx, subcluster_idx, vector_idx = item

        is_empty = self.clusters[cluster_idx][subcluster_idx].remove_(vector_idx)
        if is_empty:
            _ = self.clusters[cluster_idx].pop(subcluster_idx)
            for i, item in enumerate(self.stored_vectors):
                _, _, cluster_idx_, subcluster_idx_, vector_idx_ = item
                if cluster_idx == cluster_idx_:
                    if subcluster_idx <= subcluster_idx_:
                        self.stored_vectors[i][3] -= 1
            # self.check_integrity()
        else:
            for i, item in enumerate(self.stored_vectors):
                _, _, cluster_idx_, subcluster_idx_, vector_idx_ = item
                if cluster_idx == cluster_idx_:
                    if subcluster_idx == subcluster_idx_:
                        if vector_idx <= vector_idx_:
                            self.stored_vectors[i][4] -= 1
            # self.check_integrity()