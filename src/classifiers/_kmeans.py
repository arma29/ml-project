from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
import numpy as np
import random
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
import math


class KMeansCustom(TransformerMixin, ClusterMixin, BaseEstimator):

    def __init__(self, n_clusters=8, init='k-means++', n_init=10):
        init_lst = ['Rand-P','Rand-C','Maxmin', 'kmeans++', 'Bradley','Sorting','Projection','Luxburg','Split']
        if(init not in init_lst):
            raise ValueError("Erro")

        self.init = init
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = 300
        self.cluster_centers_ = np.array([])
        self.labels_ = np.array([])
        self.inertia_ = 9999999
        self._dict = {}
        self.__init_dict()

    def fit(self, X):
        return self._dict[self.init](X)
    
    def __init_dict(self):
        self._dict['Rand-P'] = self.__rand_p_init
        self._dict['Rand-C'] = self.__rand_c_init
        self._dict['Maxmin'] = None
        self._dict['kmeans++'] = self.__kmeans_plus_plus_init
        self._dict['Bradley'] = None
        self._dict['Sorting'] = None
        self._dict['Projection'] = None
        self._dict['Luxburg'] = None
        self._dict['Split'] = None

    def __rand_p_init(self, X):
        # Every point is put into a randomly chosen cluster and their
        # centroids are then calculated
        self.labels_ = random.choices(range(self.n_clusters), k=len(X))
        self.cluster_centers_ = list(range(self.n_clusters))
        self.__update_step(X)

        # for i in range(len(self.cluster_centers_)):
        #     if(np.isnan(self.cluster_centers_[i])):
        #         self.cluster_centers_[i] = np.zeros(self.n_clusters)


        for i in range(self.max_iter):
            # Reassign points to its closest center
            dists = euclidean_distances(X, self.cluster_centers_)
            self.labels_ = [np.argmin(x) for x in dists]

            self.__update_step(X)

            # Calculate inertia - End condition
            curr_inertia_ = sum([min(x)**2 for x in dists])
            if(self.inertia_ == curr_inertia_):
                print(f'Best inertia: {self.inertia_} , iteration: {i+1}')
                break
            self.inertia_ = curr_inertia_

        return self

    def __rand_c_init(self, X):
        # select n_clusters random points as centroids
        self.cluster_centers_ = X[random.sample(range(len(X)), self.n_clusters)]
        for i in range(self.max_iter):
            # Reassign points to its closest center
            dists = euclidean_distances(X, self.cluster_centers_)
            self.labels_ = [np.argmin(x) for x in dists]

            self.__update_step(X)

            # Calculate inertia - End condition
            curr_inertia_ = sum([min(x)**2 for x in dists])
            if(self.inertia_ == curr_inertia_):
                print(f'Best inertia: {self.inertia_} , iteration: {i+1}')
                break
            self.inertia_ = curr_inertia_

        return self
    
    def __update_step(self, X):
        # Update cluster_centers_ by mean
        for j in range(self.n_clusters):
            filter_class = [X[idx] for idx, lbl in enumerate(self.labels_) if lbl == j]

            # Check if the cluster is empty
            if(len(filter_class) == 0):
                self.cluster_centers_[j] = np.full(X.shape[1],np.inf)
            else:
                self.cluster_centers_[j] = np.mean(filter_class, axis=0)

    def __kmeans_plus_plus_init(self,X):
        kmn = KMeans(n_clusters=self.n_clusters,init='k-means++',n_init=1)
        return kmn.fit(X)

    def __pre_compute_CI_value(self, pred_centers, real_centers):
        q_arr = []
        for c in pred_centers:
            q_arr.append(np.argmin([np.linalg.norm(c-rc) for rc in real_centers]))

        sum_orphan = 0
        for j in range(len(real_centers)):
            if(j not in q_arr):
                sum_orphan = sum_orphan + 1
        return sum_orphan
    
    def compute_CI_value(self, real_centers):
        CI1 = self.__pre_compute_CI_value(self.cluster_centers_, real_centers)
        CI2 = self.__pre_compute_CI_value(real_centers, self.cluster_centers_)
        return np.max([CI1,CI2])
        

