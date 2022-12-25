import numpy as np
from links_cluster import LinksCluster, Subcluster
from scipy.spatial.distance import cosine
import itertools

class EvictingCacheWrapper(LinksCluster):
    def __init__(self, cluster_similarity_threshold: float, subcluster_similarity_threshold: float,
                 pair_similarity_maximum: float, query_whole_cluster: bool):
        super().__init__(cluster_similarity_threshold, subcluster_similarity_threshold, pair_similarity_maximum)
        self.stored_vectors = [] # [new_key, new_vector, cluster_idx, subcluster_idx, vector_idx] # TODO Update
        self.query_whole_cluster = query_whole_cluster

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

    def update_cluster(self, cl_idx: int, sc_idx: int):
        super().update_cluster(cl_idx, sc_idx)
        raise NotImplementedError("Need to update self.stored_vectors")

    def push(self, new_key: int, new_vector: np.ndarray, top_n=10) -> list:
        cluster_idx, subcluster_idx, vector_idx = super().predict_subcluster(new_vector)
        self.stored_vectors.append([new_key, new_vector, cluster_idx, subcluster_idx, vector_idx])
        return list(itertools.islice(self.query_vector(-1), top_n))

    def pop(self, stored_vectors_idx=0):
        item = self.stored_vectors[stored_vectors_idx]
        _, target_vector, cluster_idx, subcluster_idx, vector_idx = item
        self.clusters[cluster_idx][subcluster_idx].remove(vector_idx)
        # self.update_cluster(cluster_idx, subcluster_idx)