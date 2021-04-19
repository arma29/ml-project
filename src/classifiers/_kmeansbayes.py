
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from yellowbrick.cluster import KElbowVisualizer


class KMeansBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, target_names):
        super().__init__()
        self.target_names = target_names
        # self.__pos_class = target_names[1]
        self.__is_fitted = False
        # self.__dist_dict = {}
        # self.__mean_dict = {}
        # self.__std_dict = {}
        # self.bayes_threshold = 0
        # self.__exc_set = set([])
        self.best_k = []
        self.__kmeans = []

    def fit(self, X, y):
        self.__is_fitted = True

        X_neg = np.array([x[0]
                          for x in zip(X, y) if x[1] == self.target_names[0]])
        X_pos = np.array([x[0]
                          for x in zip(X, y) if x[1] == self.target_names[1]])

        # Para todas as classes
        self.best_k.append(self.__get_best_k(X_neg))
        self.__kmeans.append(
            KMeans(n_clusters=self.best_k[0], random_state=1).fit(X_neg))

        self.best_k.append(self.__get_best_k(X_pos))
        self.__kmeans.append(
            KMeans(n_clusters=self.best_k[1], random_state=1).fit(X_pos))

        X_new = np.concatenate((X_neg, X_pos))

        y_pos = np.array([y + self.best_k[0]
                          for y in self.__kmeans[1].labels_])
        y_new = np.concatenate((self.__kmeans[0].labels_, y_pos))

        self.__gaussian = GaussianNB().fit(X_new, y_new)
        return self

    def __get_best_k(self, X):
        fig, ax = plt.subplots()
        model = KMeans(random_state=1)
        v2 = KElbowVisualizer(
            model, k=(2, 7), metric='silhouette', ax=ax).fit(X)
        plt.close()  # necessário para não mostrar os gráficos do visualizer
        return 2 if v2.elbow_value_ is None else v2.elbow_value_

    def predict(self, X):
        if(not self.__is_fitted):
            raise Exception('Not fitted')

        y_pred = self.__gaussian.predict(X)
        return np.array([self.target_names[0] if y < self.best_k[0] else self.target_names[1] for y in y_pred])

    def score(self, X, y):
        return self.__accuracy_score(y, self.predict(X))

    def __accuracy_score(self, y_true, y_pred):
        if(len(y_true) != len(y_pred)):
            raise Exception('Diff lens')
        hits = [True for (a, b) in zip(y_true, y_pred) if a == b]
        return len(hits)/len(y_true)

    def predict_proba(self, X):
        predict_lst = self.predict(X)
        return np.array([[1, 0] if y == self.target_names[0] else [0, 1] for y in predict_lst])
