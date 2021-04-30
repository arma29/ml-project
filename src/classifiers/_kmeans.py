import math

import numpy as np
import sklearn.utils.validation as val
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics.pairwise import euclidean_distances


class KMeansCustom(TransformerMixin, ClusterMixin, BaseEstimator):

    def __init__(self, n_clusters=8, init='k-means++', n_init=10, real_centers=None, random_state=None):
        init_lst = ['Rand-P', 'Rand-C', 'Maxmin', 'kmeans++',
                    'Bradley', 'Sorting', 'Projection', 'Luxburg', 'Split']
        if init not in init_lst:
            raise ValueError(
                f'"{init}" é um método de inicialização desconhecido')

        self.init = init
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.real_centers = real_centers
        self.random_state = val.check_random_state(random_state)
        self.max_iter = 300
        self._dict = {}
        self.__init_dict()

    def fit(self, X):
        best_inertia = None
        for _ in range(self.n_init):
            self._dict[self.init](X)
            init = self.cluster_centers_
            kmn = KMeans(n_clusters=self.n_clusters, init=init,
                         n_init=1, algorithm='full').fit(X)

            if(best_inertia is None or kmn.inertia_ < best_inertia):
                best_inertia = kmn.inertia_
                best_labels = kmn.labels_
                best_centers = kmn.cluster_centers_
                best_iter = kmn.n_iter_
                best_i_CI_value = self.__compute_max_CI_value(
                    init, self.real_centers)
                best_f_CI_value = self.__compute_max_CI_value(
                    kmn.cluster_centers_, self.real_centers)

        self.inertia_ = best_inertia
        self.labels_ = best_labels
        self.cluster_centers_ = best_centers
        self.n_iter_ = best_iter
        self.initial_CI_value = best_i_CI_value
        self.final_CI_value = best_f_CI_value

        return self

    def __init_dict(self):
        self._dict['Rand-P'] = self.__rand_p_init
        self._dict['Rand-C'] = self.__rand_c_init
        self._dict['Maxmin'] = self.__maxmin_init
        self._dict['kmeans++'] = self.__kmeans_plus_plus_init
        self._dict['Bradley'] = self.__bradley_init
        self._dict['Sorting'] = self.__sorting_init
        self._dict['Projection'] = self.__projection_init
        self._dict['Luxburg'] = self.__luxburg_init
        self._dict['Split'] = self.__split_init

    def __rand_p_init(self, X):
        # Every point is put into a randomly chosen cluster and their
        # centroids are then calculated
        labels = self.random_state.choice(
            range(0, self.n_clusters), size=X.shape[0])
        means = [X[labels == lbl].mean(axis=0)
                 for lbl in range(self.n_clusters)]
        self.cluster_centers_ = np.vstack(means)

    def __rand_c_init(self, X):
        # No artigo - We use shuffling method
        # by swapping the position of every data point with another
        # randomly chosen point. This takes O( N ) time.
        # select n_clusters random points as centroids
        seeds = self.random_state.permutation(X.shape[0])[:self.n_clusters]
        self.cluster_centers_ = X[seeds]

    def __maxmin_init(self, X):
        n_samples, n_features = X.shape
        centers = np.empty((self.n_clusters, n_features), dtype=X.dtype)

        # Select a random centroid
        center_id = self.random_state.randint(n_samples)

        centers[0] = X[center_id]
        closest_dist_sq = np.full(
            shape=(1, n_samples),
            fill_value=999999.0,
            dtype='float64'
        )

        # At each step, the next centroid is the point that is
        # furthest (max) from its nearest (min) existing centroid
        for i in range(0, self.n_clusters - 1):
            new_dist_sq = euclidean_distances(
                X=centers[i, np.newaxis], Y=X,
                squared=False
            )

            # Update de min distance to its nearest centroid for each data in X
            np.minimum(closest_dist_sq, new_dist_sq, out=closest_dist_sq)
            next_center_id = np.argmax(closest_dist_sq)
            centers[i+1] = X[next_center_id]

        self.cluster_centers_ = centers

    def __kmeans_plus_plus_init(self, X):
        # Initial centers
        # It chooses the first centroid randomly and the next ones
        # using a weighted probability p i = cost i /SUM( cost i ), where cost i is
        # the squared distance of the data point x i to its nearest centroids.
        self.cluster_centers_, _ = kmeans_plusplus(
            X, n_clusters=self.n_clusters)

    def __sorting_init(self, X):
        n_samples, _ = X.shape

        # Select a random point
        center_id = self.random_state.randint(n_samples)
        center = X[center_id]

        # Order the X array relative to the distance to the random point selected
        closest_dists = euclidean_distances(
            X=center.reshape(1, -1),
            Y=X
        )
        ordered = np.argsort(closest_dists).reshape(-1,)

        # The centroids are then selected as every N / k th point in this order
        indices = self.__get_nkth_indices(ordered, n_samples)
        self.cluster_centers_ = X[indices]

    def __projection_init(self, X):
        n_samples, _ = X.shape

        # The heuristic takes two
        # random data points and projects to the line passing by these two
        # reference points.
        seeds = self.random_state.permutation(n_samples)[:2]
        p1, p2 = X[seeds]

        # Sort points according with projection point multiplier
        ordered = np.argsort([self.__get_projection(x, p1, p2) for x in X])

        # The centroids are then selected as every N / k th point in this order
        indices = self.__get_nkth_indices(ordered, n_samples)
        self.cluster_centers_ = X[indices]

    def __get_projection(self, x, p1, p2):
        return np.dot((p2 - p1), (x - p1)) / np.dot((p2 - p1), (p2 - p1))

    def __split_init(self, X):
        # Split algorithm puts all points into a single cluster, and then it-
        # eratively splits one cluster at a time until k clusters are reached.
        partitions = [[]]*self.n_clusters
        partitions[0] = np.arange(X.shape[0])

        # In this paper, we therefore implement a simpler variant. We
        # always select the biggest cluster to be split. The split is done by
        # two random points in the cluster.
        for i in range(1, self.n_clusters):
            biggest_idx, biggest_partition = max(
                enumerate(partitions),
                key=lambda x: len(x[1])
            )

            # K-means is then applied but only within the cluster that was split as done in
            kmn = KMeans(n_clusters=2, n_init=1, init='random',
                         algorithm='full').fit(X[biggest_partition])

            partitions[biggest_idx] = biggest_partition[np.where(kmn.labels_ == 0)[
                0]]
            partitions[i] = biggest_partition[np.where(kmn.labels_ == 1)[0]]

        self.cluster_centers_ = np.array(
            [np.mean(X[part], axis=0) for part in partitions])

    def __bradley_init(self, X):
        # Sub-sample of size N/R where R = 10
        R = 10
        seeds = self.random_state.permutation(X.shape[0])
        randomized = X[seeds]
        partitions = np.array_split(randomized, R)

        # However, instead of taking the best clustering of the repeats, a new dataset
        # is created from the R*k centroids.
        kmn = KMeans(n_clusters=self.n_clusters,
                     init='random', n_init=1, algorithm='full')
        rk_centroids = np.vstack(
            [kmn.fit(part).cluster_centers_ for part in partitions])

        # This new dataset is then clustered by repeated k-means ( R repeats).
        kmn = KMeans(n_clusters=self.n_clusters, init='random',
                     n_init=R, algorithm='full').fit(rk_centroids)

        self.cluster_centers_ = kmn.cluster_centers_

    def __luxburg_init(self, X):
        # Luxburg [50] first selects k ∗SQRT( k ) preliminary clusters using
        # k-means and then eliminates the smallest ones.
        L = math.floor(self.n_clusters * math.sqrt(self.n_clusters))
        kmn = KMeans(n_clusters=L, init='random',
                     n_init=1, algorithm='full').fit(X)
        sizes = [len(kmn.labels_[kmn.labels_ == i]) for i in range(L)]
        limit = np.mean(sizes)
        self.cluster_centers_ = kmn.cluster_centers_[
            np.where(sizes >= limit)[0]]

        # (4) After this, the furthest point heuristic is used to select the k clusters
        # from the preliminary set of clusters. - same as maxmin
        self.__maxmin_init(self.cluster_centers_)

    def __compute_CI_value(self, pred_centers, real_centers):
        # The ideal: every pred_center is mapped to a different real_center, CI = 0
        # Otherwise: the result are the missing real_centers

        # closest_centers[0] = distância de X[0] para cada ponto em Y
        # closest_centers[-1] = distância de X[-1] para cada ponto em Y
        closest_centers = np.argmin(euclidean_distances(
            X=pred_centers,
            Y=real_centers
        ), axis=0)

        return self.n_clusters - np.unique(closest_centers).shape[0]

    def __compute_max_CI_value(self, pred_centers, real_centers):
        # Assign to field
        CI1 = self.__compute_CI_value(pred_centers, real_centers)
        CI2 = self.__compute_CI_value(real_centers, pred_centers)
        return np.max([CI1, CI2])

    def __get_nkth_indices(self, array, N):
        size = int(
            N/self.n_clusters) if N % self.n_clusters == 0 else math.ceil(N/self.n_clusters)
        return array[::size]

    def initialize_kmeans(self, X):
        # Measure init time
        self._dict[self.init](X)
        return self
