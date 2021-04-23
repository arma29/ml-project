from numpy.core.defchararray import partition
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
import numpy as np
import random
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
import math
from sklearn.cluster import kmeans_plusplus


class KMeansCustom(TransformerMixin, ClusterMixin, BaseEstimator):

    def __init__(self, n_clusters=8, init='k-means++', n_init=10, real_centers=None):
        init_lst = ['Rand-P', 'Rand-C', 'Maxmin', 'kmeans++',
                    'Bradley', 'Sorting', 'Projection', 'Luxburg', 'Split']
        if init not in init_lst:
            raise ValueError(f'"{init}" é um método de inicialização desconhecido')

        self.init = init
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.real_centers = real_centers
        self.initial_CI_value = -1
        self.final_CI_value = -1
        self.max_iter = 300
        self.iterations = 0
        self.cluster_centers_ = []
        self.labels_ = []
        self.inertia_ = 9999999
        self._dict = {}
        self.__init_dict()

    def fit(self, X):
        best_inertia = np.inf
        best_labels = None
        best_centers = None
        best_iter = None
        best_i_CI_value = None
        best_f_CI_value = None
        for _ in range(self.n_init):
            self.__kmeans_algorithm(X, self._dict[self.init])
            if(self.inertia_ < best_inertia):
                best_inertia = self.inertia_
                best_labels = self.labels_
                best_centers = self.cluster_centers_
                best_iter = self.iterations
                best_i_CI_value = self.initial_CI_value
                best_f_CI_value = self.final_CI_value

        self.inertia_ = best_inertia
        self.labels_ = best_labels
        self.cluster_centers_ = best_centers
        self.iterations = best_iter
        self.initial_CI_value = best_i_CI_value
        self.final_CI_value = best_f_CI_value

        # print(f'Best inertia: {self.inertia_} , iteration: {self.iterations}, {self._dict[self.init].__name__}')

        return self

    def __kmeans_algorithm(self, X, init_func):
        init_func(X)
        self.initial_CI_value = self.__compute_max_CI_value()

        for i in range(self.max_iter):

            dists = self.__kmeans_assign_step(X)

            self.__kmeans_update_step(X,self.n_clusters)

            # Calculate inertia - End condition
            curr_inertia_ = sum([min(x)**2 for x in dists])
            if(self.inertia_ == curr_inertia_):
                self.iterations = i+1
                self.final_CI_value = self.__compute_max_CI_value()
                break
            self.inertia_ = curr_inertia_

        return self

    def __kmeans_assign_step(self, X):
        dists = euclidean_distances(X, self.cluster_centers_)
        self.labels_ = np.array([np.argmin(x) for x in dists])
        return dists

    def __kmeans_update_step(self, X, count):
        # Update cluster_centers_ by mean
        clusters = [X[self.labels_ == lbl] for lbl in range(count)]
        # Um cluster vazio tera como centro um ponto que não interfira nos cálculos
        means = [c.mean(axis=0) if c.size != 0 else [999999.0] * X.shape[1] for c in clusters]
        self.cluster_centers_ = np.vstack(means)

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
        # labels = random.choices(range(0, self.n_clusters), k=X.shape[0])
        labels = np.random.choice(range(0, self.n_clusters), size=X.shape[0])
        means = [X[labels == lbl].mean(axis=0)
                 for lbl in range(self.n_clusters)]
        self.cluster_centers_ = np.vstack(means)

    def __rand_c_init(self, X):
        # No artigo - We use shuffling method
        # by swapping the position of every data point with another
        # randomly chosen point. This takes O( N ) time.
        # select n_clusters random points as centroids
        self.cluster_centers_ = X[random.sample(
            range(X.shape[0]), k=self.n_clusters)]

    def __maxmin_init(self, X):
        # TODO: Need improvement

        # Select a random centroid
        self.cluster_centers_ = [random.choice(X)]
        nearest_centroid_arr = [999999.0]*X.shape[0]
        max_dist = -1
        next_centroid_idx = -1

        # At each step, the next centroid is the point that is
        # furthest (max) from its nearest (min) existing centroid
        for _ in range(self.n_clusters - 1):
            for idx in range(X.shape[0]):
                dist = np.linalg.norm(X[idx] - self.cluster_centers_[-1])
                if(dist < nearest_centroid_arr[idx]):
                    nearest_centroid_arr[idx] = dist

                if(nearest_centroid_arr[idx] > max_dist):
                    max_dist = nearest_centroid_arr[idx]
                    next_centroid_idx = idx

            self.cluster_centers_.append(X[next_centroid_idx])
            max_dist = -1
            next_centroid_idx = -1

    def __kmeans_plus_plus_init(self, X):
        # Initial centers
        # It chooses the first centroid randomly and the next ones
        # using a weighted probability p i = cost i /SUM( cost i ), where cost i is
        # the squared distance of the data point x i to its nearest centroids.
        self.cluster_centers_, _ = kmeans_plusplus(
            X, n_clusters=self.n_clusters)

    def __sorting_init(self, X):
        # Select a random point
        center = random.choice(X)
        # Order the X array relative to the distance to the random point selected
        ordered = np.array(sorted(X, key=lambda x: np.linalg.norm(x - center)))
        # The centroids are then selected as every N / k th point in this order
        indices = self.__get_nkth_indices(X, self.n_clusters)
        self.cluster_centers_ = ordered[indices]

    def __projection_init(self, X):
        # The heuristic takes two
        # random data points and projects to the line passing by these two
        # reference points.
        p1, p2 = X[random.sample(range(X.shape[0]), k=2)]

        # # Sort points according with projection point multiplier
        # ordered_idx = np.array(sorted(
        #     list(range(X.shape[0])), key=lambda i: self.__get_projection(X[i], p1, p2)))

        # # Equaly divide the k clusters
        # partitions = self.__generate_partitions(ordered_idx, self.n_clusters)

        # self.labels_ = [-1]*X.shape[0]
        # for i in range(self.n_clusters):
        #     for elem in partitions[i]:
        #         self.labels_[elem] = i

        # self.cluster_centers_ = [-1]*self.n_clusters
        # self.__update_step(X)

        # Sort points according with projection point multiplier
        ordered = np.array(
            sorted(X, key=lambda x: self.__get_projection(x, p1, p2)))
        # The centroids are then selected as every N / k th point in this order
        indices = self.__get_nkth_indices(X, self.n_clusters)
        self.cluster_centers_ = ordered[indices]

    def __get_projection(self, x, p1, p2):
        return np.dot((p2 - p1), (x - p1)) / np.dot((p2 - p1), (p2 - p1))

    def __split_init(self, X):
        # Split algorithm puts all points into a single cluster, and then it-
        # eratively splits one cluster at a time until k clusters are reached.
        partitions = [np.array(range(X.shape[0]))]

        # In this paper, we therefore implement a simpler variant. We
        # always select the biggest cluster to be split. The split is done by
        # two random points in the cluster.
        for _ in range(self.n_clusters - 1):
            biggest_partition = sorted(
                enumerate(partitions), key=lambda x: len(x[1]), reverse=True)[0]

            kmn = KMeans(n_clusters=2, n_init=1, init='random').fit(
                X[biggest_partition[1]])

            clusterA = []
            clusterB = []
            for idx, lbl in enumerate(kmn.labels_):
                if(lbl == 0):
                    clusterA.append(biggest_partition[1][idx])
                elif(lbl == 1):
                    clusterB.append(biggest_partition[1][idx])

            del partitions[biggest_partition[0]]
            partitions.append(clusterA)
            partitions.append(clusterB)

        self.cluster_centers_ = [np.mean(X[part], axis=0)
                                 for part in partitions]

    def __bradley_init(self, X):
        # TODO: Need improvement

        # Sub-sample of size N/R where R = 10
        R = 10
        randomized = X[random.sample(range(X.shape[0]), k=X.shape[0])]
        partitions = self.__generate_partitions(randomized, R)

        # However, instead of taking the best clustering of the repeats, a new dataset
        # is created from the R*k centroids.
        rk_centroids = [KMeans(n_clusters=self.n_clusters, init='random', n_init=1).fit(
            part).cluster_centers_ for part in partitions]

        concat_centroids = rk_centroids[0]
        for i in range(1, R):
            concat_centroids = np.concatenate(
                [concat_centroids, rk_centroids[i]])

        # This new dataset is then clustered by repeated k-means ( R repeats).
        kmn = KMeans(n_clusters=self.n_clusters, init='random',
                     n_init=R).fit(concat_centroids)

        self.cluster_centers_ = kmn.cluster_centers_

    def __luxburg_init(self, X):
        # (1) Select L preliminary centers uniformly at random from the
        # given data set, where L ≈ K log(K). (No artigo usa-se k*sqrt(k))
        L = math.floor(self.n_clusters * math.sqrt(self.n_clusters))
        randomized = X[random.sample(range(X.shape[0]), k=X.shape[0])]
        indices = self.__get_nkth_indices(X, L)
        self.cluster_centers_ = randomized[indices]

        # (2) Run one step of K-means, that is assign the data points to
        # the preliminary centers and re-adjust the centers once.
        _ = self.__kmeans_assign_step(randomized)
        self.__kmeans_update_step(randomized,L)

        # (3) Remove all centers for which the mass of the assigned data
        # points is smaller than p0 ≈ 1/L. (No artigo elimina-se os menores clusters)
        selected_centers = []
        for i in range(L):
            curr_indices = [idx for idx, lbl in enumerate(
                self.labels_) if lbl == i]
            if(len(curr_indices) > math.ceil(math.sqrt(self.n_clusters))):
                selected_centers.append(self.cluster_centers_[i])

        # (4) After this, the furthest point heuristic is used to select the k clusters
        # from the preliminary set of clusters. - same as maxmin
        self.__maxmin_init(np.array(selected_centers))

    def __compute_CI_value(self, pred_centers, real_centers):
        q_arr = []
        for c in pred_centers:
            q_arr.append(np.argmin([np.linalg.norm(c-rc)
                                    for rc in real_centers]))

        sum_orphan = 0
        for j in range(len(real_centers)):
            if(j not in q_arr):
                sum_orphan = sum_orphan + 1
        return sum_orphan

    def __compute_max_CI_value(self):
        # Assign to field
        CI1 = self.__compute_CI_value(self.cluster_centers_, self.real_centers)
        CI2 = self.__compute_CI_value(self.real_centers, self.cluster_centers_)
        return np.max([CI1, CI2])

    def __generate_partitions(self, array, count):
        k_size = math.floor((len(array)/count) + 0.5)
        partitions = [array[i:i+k_size] for i in range(0, len(array), k_size)]
        return partitions

    def __get_nkth_indices(self, X, count):
        indices = [math.floor(i*(X.shape[0]/count) + 0.5)
                   for i in range(count)]
        return indices

    def initialize_kmeans(self, X):
        # Measure init time
        self._dict[self.init](X)
