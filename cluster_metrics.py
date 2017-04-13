from __future__ import division
from collections import defaultdict

from math import sqrt, log

import numpy as np

from scipy.spatial.distance import cdist, pdist

from sklearn.utils import check_random_state


def inertia(X, labels, metric='sqeuclidean', p=2):
    """
    Given data and their cluster assigment, compute the sum of
    within-cluster mean distance to cluster's mean

    Parameter
    ---------
    X: numpy array of shape (nb_data, nb_feature)
    labels: list of int of length nb_data
    metric: a string naming a scipy.spatial distance. can be in
        ['euclidian', 'minkowski', 'seuclidiean', 'sqeuclidean', 'chebyshev'
         'cityblock', 'cosine', 'correlation', 'hamming', 'jaccard',
         'Bray-Curtis', 'mahalanobis', 'yule', 'matching', 'dice', 'kulsinski',
         'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath',
         'canberra', 'wminkowski'])
    p : double
        The p-norm to apply (for Minkowski, weighted and unweighted)

    Return
    ------
    distortion: float
    """
    if metric == 'l2':
        # Translate to something understood by scipy
        metric = 'euclidean'
    elif metric in ('l1', 'manhattan'):
        metric = 'cityblock'

    assi = defaultdict(list)
    for i, l in enumerate(labels):
        assi[l].append(i)

    inertia = .0
    nb_feature = X.shape[1]
    for points in assi.values():
        clu_points = X[points, :]
        clu_center = np.mean(clu_points, axis=0).reshape(1, nb_feature)
        inertia += (np.sum(cdist(clu_points, clu_center, metric=metric, p=p)) /
                    (2 * len(clu_points)))
    return inertia


def normal_inertia(X, cluster_estimator, nb_draw=100,
                   metric='sqeuclidean', p=2, random_state=None,
                   mu=None, sigma=None):
    """
    Draw multivariate normal data of size data_shape = (nb_data, nb_feature),
    with same mean and covariance as X.
    Clusterize data using cluster_estimator and compute inertia

    Parameter
    ---------
    X numpy array of size (nb_data, nb_feature)
    cluster_estimator: ClusterMixing estimator object.
        need parameter n_clusters
        need method fit_predict: X -> labels
    nb_draw: number of samples to calculate expected_inertia
    metric: a string naming a scipy.spatial distance. can be in
        ['euclidian', 'minkowski', 'seuclidiean', 'sqeuclidean', 'chebyshev'
         'cityblock', 'cosine', 'correlation', 'hamming', 'jaccard',
         'Bray-Curtis', 'mahalanobis', 'yule', 'matching', 'dice', 'kulsinski',
         'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath',
         'canberra', 'wminkowski'])
    p : double
        The p-norm to apply (for Minkowski, weighted and unweighted)
    mu: mean of drawn data
    sigma: covariance matrix of drawn data

    Return
    ------
    dist: list of inertias (float) obtained on random dataset
    """
    rng = check_random_state(random_state)
    nb_data, nb_feature = X.shape

    if mu is None:
        # data mean has no influence on distortion
        mu = np.zeros(nb_feature)
    if sigma is None:
        sigma = np.cov(X.transpose())

    dist = []
    for i in range(nb_draw):
        X_rand = rng.multivariate_normal(mu, sigma, size=nb_data)
        dist.append(inertia(
            X_rand, cluster_estimator.fit_predict(X_rand),
            metric, p))

    return dist


def uniform_inertia(X, cluster_estimator, nb_draw=100, val_min=None,
                    val_max=None, metric='sqeuclidean', p=2,
                    random_state=None):
    """
    Uniformly draw data of size data_shape = (nb_data, nb_feature)
    in the smallest hyperrectangle containing real data X.
    Clusterize data using cluster_estimator and compute inertia

    Parameter
    ---------
    X: numpy array of shape (nb_data, nb_feature)
    cluster_estimator: ClusterMixing estimator object.
        need parameter n_clusters
        need method fit_predict: X -> labels
    nb_draw: number of samples to calculate expected_inertia
    val_min: minimum values of each dimension of input data
        array of length nb_feature
    val_max: maximum values of each dimension of input data
        array of length nb_feature
    metric: a string naming a scipy.spatial distance. can be in
        ['euclidian', 'minkowski', 'seuclidiean', 'sqeuclidean', 'chebyshev'
         'cityblock', 'cosine', 'correlation', 'hamming', 'jaccard',
         'Bray-Curtis', 'mahalanobis', 'yule', 'matching', 'dice', 'kulsinski',
         'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath',
         'canberra', 'wminkowski'])
    p : double
        The p-norm to apply (for Minkowski, weighted and unweighted)

    Return
    ------
    dist: list of distortions (float) obtained on random dataset
    """
    rng = check_random_state(random_state)
    if val_min is None:
        val_min = np.min(X, axis=0)
    if val_max is None:
        val_max = np.max(X, axis=0)

    dist = []
    for i in range(nb_draw):
        X_rand = rng.uniform(size=X.shape) * (val_max - val_min) + val_min
        dist.append(inertia(X_rand, cluster_estimator.fit_predict(X_rand),
                            metric, p))

    return dist


def gap_statistic(X, cluster_estimator, k_max=None, nb_draw=100,
                  random_state=None, draw_model='uniform',
                  metric='sqeuclidean', p=2):
    """
    Estimating optimal number of cluster for data X with cluster_estimator by
    comparing inertia of clustered real data with inertia of clustered
    random data. Let W_rand(k) be the inertia of random data in k clusters,
    W_real(k) inertia of real data in k clusters, statistic gap is defined
    as

    Gap(k) = E(log(W_rand(k))) - log(W_real(k))

    We draw nb_draw random data "shapened-like X" (shape depend on draw_model)
    We select the smallest k such as the gap between inertia of k clusters
    of random data and k clusters of real data is superior to the gap with
    k + 1 clusters minus a "standard-error" safety. Precisely:

    k_star = min_k k
         s.t. Gap(k) >= Gap(k + 1) - s(k + 1)
              s(k) = stdev(log(W_rand)) * sqrt(1 + 1 / nb_draw)

    From R.Tibshirani, G. Walther and T.Hastie, Estimating the number of
    clusters in a dataset via the Gap statistic, Journal of the Royal
    Statistical Socciety: Seris (B) (Statistical Methodology), 63(2), 411-423

    Parameter
    ---------
    X: data. array nb_data * nb_feature
    cluster_estimator: ClusterMixing estimator object.
        need parameter n_clusters
    nb_draw: int: number of random data of shape (nb_data, nb_feature) drawn
        to estimate E(log(D_rand(k)))
    draw_model: under which i.i.d data are draw. default: uniform data
        (following Tibshirani et al.)
        can be 'uniform', 'normal' (Gaussian distribution)
    metric: a string naming a scipy.spatial distance. can be in
        ['euclidian', 'minkowski', 'seuclidiean', 'sqeuclidean', 'chebyshev'
         'cityblock', 'cosine', 'correlation', 'hamming', 'jaccard',
         'Bray-Curtis', 'mahalanobis', 'yule', 'matching', 'dice', 'kulsinski',
         'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath',
         'canberra', 'wminkowski'])
    p : double
        The p-norm to apply (for Minkowski, weighted and unweighted)

    Return
    ------
    k: int: number of cluster that maximizes the gap statistic
    """
    rng = check_random_state(random_state)

    # if no maximum number of clusters set, take datasize divided by 2
    if not k_max:
        k_max = X.shape[0] // 2
    if draw_model == 'uniform':
        val_min = np.min(X, axis=0)
        val_max = np.max(X, axis=0)
    elif draw_model == 'normal':
        mu = np.mean(X, axis=0)
        sigma = np.cov(X.transpose())

    old_gap = - float("inf")
    for k in range(2, k_max + 2):
        cluster_estimator.set_params(n_clusters=k)
        real_dist = inertia(X, cluster_estimator.fit_predict(X),
                            metric, p)
        # expected distortion
        if draw_model == 'uniform':
            rand_dist = uniform_inertia(X, cluster_estimator, nb_draw,
                                        val_min, val_max, metric,
                                        p)
        elif draw_model == 'normal':
            rand_dist = normal_inertia(X, cluster_estimator, nb_draw,
                                       metric=metric,
                                       p=p, mu=mu, sigma=sigma)
        else:
            raise ValueError(
                "For gap statistic, model for random data is unknown")
        rand_dist = np.log(rand_dist)
        exp_dist = np.mean(rand_dist)
        std_dist = np.std(rand_dist)
        gap = exp_dist - log(real_dist)
        safety = std_dist * sqrt(1 + 1 / nb_draw)
        if k > 2 and old_gap >= gap - safety:
            return k - 1
        old_gap = gap
    # if k was found, the function would have returned
    # no clusters were found -> only 1 cluster
    return 1


def adjacency_matrix(cluster_assignement):
    """
    Parameter
    ---------
    cluster_assignement: vector (n_samples) of int i, 0 <= i < k

    Return
    ------
    adj_matrix: matrix (n_samples, n_samples)
        adji_matrix[i, j] = cluster_assignement[i] == cluster_assignement[j]
    """
    n_samples = len(cluster_assignement)
    adj_matrix = np.zeros((n_samples, n_samples))
    for i, val in enumerate(cluster_assignement):
        for j in range(i, n_samples):
            linked = val == cluster_assignement[j]
            adj_matrix[i, j] = linked
            adj_matrix[j, i] = linked
    return adj_matrix


def fowlkes_mallows_index(clustering_1, clustering_2):
    """
    Mesure the similarity of two clusterings of a set of points.
    Let:
    - TP be the number of pair of points (x_i, x_j) that belongs
        in the same clusters in both clustering_1 and clustering_2
    - FP be the number of pair of points (x_i, x_j) that belongs
        in the same clusters in clustering_1 and not in clustering_2
    - FN be the number of pair of points (x_i, x_j) that belongs
        in the same clusters in clustering_2 and not in clustering_1
    The Fowlkes-Mallows index has the following formula:
        fowlkes_mallows_index = TP / sqrt((TP + FP) * (TP + FN))
    Parameter
    ---------
    clustering_1: list of int.
        "clustering_1[i] = c" means that point i is assigned to cluster c
    clustering_2: list of int.
        "clustering_2[i] = c" means that point i is assigned to cluster c
    Return
    ------
    fowlkes_mallows_index: float between 0 and 1. 1 means that both
        clusterings perfectly match, 0 means that they totally disconnect
    """
    adj_mat_1 = adjacency_matrix(clustering_1)
    adj_mat_2 = adjacency_matrix(clustering_2)
    return 1 - cosine(adj_mat_1.flatten(), adj_mat_2.flatten())


def stability(X, cluster_estimator, k_max=None, nb_draw=100, prop_subset=.8,
              random_state=None, p=None, distance='fowlkes-mallows',
              verbose=False):
    """Stability algorithm.
    For k from 2 to k_max, compute stability of cluster estimator to produce k
    clusters. Stability measures if the estimator produces the same clusters
    given small variations in the input data. It draws two overlapping subsets
    A and B of input data. For points in the two subsets, we compute the
    clustering C_A and C_B done on subsets A and B. We then compute the
    similarity of those clustering. We can use the opposite of a distance
    as a similarity

    The stability of cluster_estimator with k cluster is the expectation of
    similarity(C_A, C_B)

    Ref: Ben-Hur, Elisseeff, Guyon: a stability based method for discovering
    structure in clusterd data, 2002
    Overview of stability: Luxburg: clustering stability: an overview

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        The observations to cluster.
    cluster_estimator: ClusterMixing estimator object.
        need parameter n_clusters
        need method fit_predict: X -> labels
    k_max: int: maximum number of clusters (default = n_samples / 2)
    nb_draw: number of draws to estimate expectation of expectation of
        similarity(C_A, C_B)
    prop_subset: 0 < float < 1: proportion of input data taken in each subset
    distance: a string naming a distance or a cluster similarity. can be in
        ['euclidian', 'minkowski', 'seuclidiean', 'sqeuclidean', 'chebyshev'
         'cityblock', 'cosine', 'correlation', 'hamming', 'jaccard',
         'Bray-Curtis', 'mahalanobis', 'yule', 'matching', 'dice', 'kulsinski',
         'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath',
         'canberra', 'wminkowski', 'fowlkes-mallows'])
    p : double
        The p-norm to apply (for Minkowski, weighted and unweighted)
    Return
    ------
    k: int
    """
    rng = check_random_state(random_state)
    cluster_similarity = function_cluster_similarity(distance, p)

    n_samples, n_features = X.shape
    if not k_max:
        k_max = n_samples // 2

    best_stab, best_k = 0, 0
    for k in range(2, k_max + 1):
        cluster_estimator.set_params(n_clusters=k)
        this_score = sum(
            _one_stability_measure(cluster_estimator, X, prop_subset,
                                   cluster_similarity)
            for _ in range(nb_draw)) / nb_draw
        if verbose:
            print('for %d cluster, stability is %f' % (k, this_score))

        if this_score >= best_stab:
            best_stab = this_score
            best_k = k

    return best_k


def _one_stability_measure(cluster_estimator, X, prop_sample,
                           cluster_similarity, random_state=None):
    """
    Draws two subsets A and B from X, compute C_A, clustering on subset
    A, and C_B, clustering on subset B, then returns

    similarity(C_A, C_B)

    Parameter
    ---------
    X: array of size n_samples, n_features
    cluster_estimator: ClusterMixing estimator object.
        need parameter n_clusters
        need method fit_predict: X -> labels
    prop_sample: 0 < float < 1, proportion of X taken in each subset
    cluster_similarity: function (list, list) -> float
    """
    rng = check_random_state(random_state)

    n_sample = X.shape[0]
    set_1 = rng.uniform(size=n_sample) < prop_sample
    set_2 = rng.uniform(size=n_sample) < prop_sample
    nb_points_1, nb_points_2 = 0, 0
    points_1, points_2 = [], []
    common_points_1, common_points_2 = [], []
    for i, (is_1, is_2) in enumerate(zip(set_1, set_2)):
        if is_1 and is_2:
            common_points_1.append(nb_points_1)
            common_points_2.append(nb_points_2)
        if is_1:
            points_1.append(i)
            nb_points_1 += 1
        if is_2:
            points_2.append(i)
            nb_points_2 += 1

    assi_1 = cluster_estimator.fit_predict(X[np.ix_(points_1)])
    assi_2 = cluster_estimator.fit_predict(X[np.ix_(points_2)])

    clustering_1 = [assi_1[c] for c in common_points_1]
    clustering_2 = [assi_2[c] for c in common_points_2]
    return cluster_similarity(clustering_1, clustering_2)


def function_cluster_similarity(metric='fowlkes-mallows', p=None):
    """
    Given the name of a distance, return function to estimate
    two clusterings  similarity

    Parameter
    --------
    metric: a string naming a distance or a cluster similarity. can be in
        ['euclidian', 'minkowski', 'seuclidiean', 'sqeuclidean', 'chebyshev'
         'cityblock', 'cosine', 'correlation', 'hamming', 'jaccard',
         'Bray-Curtis', 'mahalanobis', 'yule', 'matching', 'dice', 'kulsinski',
         'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath',
         'canberra', 'wminkowski', 'fowlkes-mallows'])
    p : double
        The p-norm to apply (for Minkowski, weighted and unweighted)

    Return:
    function (clustering_1, clustering_2) -> similarity; with:
        clustering_k: a list. clustering_k[i] = c means that
           point x_i belongs to cluster c in clustering k
        similarity: float
    """
    if metric == 'fowlkes-mallows':
        return fowlkes_mallows_index
    if metric == 'l2':
        # Translate to something understood by scipy
        metric = 'euclidean'
    elif metric in ('l1', 'manhattan'):
        metric = 'cityblock'

    def cluster_dist(clustering_1, clustering_2):
        adj_mat_1 = adjacency_matrix(clustering_1).flatten()
        adj_mat_2 = adjacency_matrix(clustering_2).flatten()
        return -pdist([adj_mat_1, adj_mat_2], metric=metric, p=p)
    return cluster_dist


def calinski_harabaz_index(X, labels):
    """
    Compute the Calinski and Harabaz (1974). It a ratio between the
    within-cluster dispersion and the between-cluster dispersion
    CH(k) = trace(B_k) / (k -1) * (n - k) / trace(W_k)
    With B_k the between group dispersion matrix, W_k the within-cluster
    dispersion matrix
    B_k = \sum_q n_q (c_q - c) (c_q -c)^T
    W_k = \sum_q \sum_{x \in C_q} (x - c_q) (x - c_q)^T
    Ref: R.B.Calinsky, J.Harabasz: A dendrite method for cluster analysis 1974
    Parameter
    ---------
    X: numpy array of size (nb_data, nb_feature)
    labels: list of int of length nb_data: labels[i] is the cluster
        assigned to X[i, :]
    Return
    ------
    res: float: mean silhouette of this clustering
    """
    assi = defaultdict(list)
    for i, l in enumerate(labels):
        assi[l].append(i)

    nb_data, nb_feature = X.shape
    disp_intra = np.zeros((nb_feature, nb_feature))
    disp_extra = np.zeros((nb_feature, nb_feature))
    center = np.mean(X, axis=0)

    for points in assi.values():
        clu_points = X[points, :]
        # unbiaised estimate of variace is \sum (x - mean_x)^2 / (n - 1)
        # so, if I want sum of dispersion, I need
        # W_k = cov(X) * (n - 1)
        nb_point = clu_points.shape[0]
        disp_intra += np.cov(clu_points, rowvar=0) * (nb_point - 1)
        extra_var = (np.mean(clu_points, axis=0) - center).reshape(
            (nb_feature, 1))
        disp_extra += np.multiply(extra_var, extra_var.transpose()) * nb_point
    return (disp_extra.trace() * (nb_data - len(assi)) /
            (disp_intra.trace() * (len(assi) - 1)))


def calc_calinski_harabaz(X, cluster_estimator, n_clusters):
    """
    Compute calinski harabaz for clusters made by cluster estimator
    Parameter
    ---------
    X numpy array of size (nb_data, nb_feature)
    cluster_estimator: ClusterMixing estimator object.
        need parameter n_clusters
        need method fit_predict: X -> labels
    n_clusters: number of clusters
    """
    cluster_estimator.set_params(n_clusters=n_clusters)
    return calinski_harabaz_index(X, cluster_estimator.fit_predict(X))


def max_CH_index(X, cluster_estimator, k_max=None):
    """
    Select number of cluster maximizing the Calinski and Harabasz (1974).
    It a ratio between the within-cluster dispersion and the between-cluster
    dispersion
    Ref: R.B.Calinsky, J.Harabasz: A dendrite method for cluster analysis 1974
    Parameters
    ----------
    X: numpy array of shape (nb_date, nb_features)
    cluster_estimator: ClusterMixing estimator object.
        need parameter n_clusters
        need method fit_predict: X -> labels
    k_max: int: maximum number of clusters
    Return
    ------
    k_star: int: optimal number of cluster
    """
    # if no maximum number of clusters set, take datasize divided by 2
    if not k_max:
        k_max = X.shape[0] // 2

    return max((k for k in range(2, k_max + 1)),
               key=lambda k: calc_calinski_harabaz(X, cluster_estimator, k))


def distortion(X, labels, distortion_meth='sqeuclidean', p=2):
    """
    Given data and their cluster assigment, compute the distortion D
    Parameter
    ---------
    X: numpy array of shape (nb_data, nb_feature)
    labels: list of int of length nb_data
    distortion_meth: can be a function X, labels -> float,
        can be a string naming a scipy.spatial distance. can be in
        ['euclidian', 'minkowski', 'seuclidiean', 'sqeuclidean', 'chebyshev'
         'cityblock', 'cosine', 'correlation', 'hamming', 'jaccard',
         'Bray-Curtis', 'mahalanobis', 'yule', 'matching', 'dice', 'kulsinski',
         'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath',
         'canberra', 'wminkowski'])
    p : double
        The p-norm to apply (for Minkowski, weighted and unweighted)
    Return
    ------
    distortion: float
    """
    if isinstance(distortion_meth, str):
        return distortion_metrics(X, labels, distortion_meth, p)
    else:
        return distortion_meth(X, labels)


def distortion_metrics(X, labels, metric='sqeuclidean', p=2):
    """
    Given data and their cluster assigment, compute the distortion D
    D = \sum_{x \in X} distance(x, c_x)
    With c_x the center of the cluster containing x, distance is the distance
    defined by metrics
    Parameter
    ---------
    X: numpy array of shape (nb_data, nb_feature)
    labels: list of int of length nb_data
    metric: string naming a scipy.spatial distance. metric can be in
        ['euclidian', 'minkowski', 'seuclidiean', 'sqeuclidean', 'chebyshev'
         'cityblock', 'cosine', 'correlation', 'hamming', 'jaccard',
         'Bray-Curtis', 'mahalanobis', 'yule', 'matching', 'dice', 'kulsinski',
         'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath',
         'wminkowski', 'canberra']
    p : double
        The p-norm to apply (for Minkowski, weighted and unweighted)
    Return
    ------
    distortion: float
    """
    if metric == 'l2':
        # Translate to something understood by scipy
        metric = 'euclidean'
    elif metric in ('l1', 'manhattan'):
        metric = 'cityblock'

    assi = defaultdict(list)
    for i, l in enumerate(labels):
        assi[l].append(i)

    distance_sum = .0
    nb_feature = X.shape[1]
    for points in assi.values():
        clu_points = X[points, :]
        clu_center = np.mean(clu_points, axis=0).reshape(1, nb_feature)
        distance_sum += np.sum(cdist(
            clu_points, clu_center, metric=metric, p=p))

    return distance_sum / X.shape[1]


def distortion_jump(X, cluster_estimator, k_max=None,
                    distortion_meth='sqeuclidean', p=2):
    """
    Find the number of clusters that maximizes efficiency while minimizing
    error by information theoretic standards (wikipedia). For each number of
    cluster, it calculates the distortion reduction. Roughly, it selects k such
    as the difference between distortion with k clusters minus distortion with
    k-1 clusters is maximal.
    More precisely, let d(k) equals distortion with k clusters.
    Let Y=nb_feature/2, let D[k] = d(k)^{-Y}
    k^* = argmax(D[k] - D[k-1])
    Parameters
    ----------
    X: numpy array of shape (nb_date, nb_features)
    cluster_estimator: ClusterMixing estimator object.
        need parameter n_clusters
        need method fit_predict: X -> labels
    k_max: int: maximum number of clusters
    distortion_meth: can be a function X, labels -> float,
        can be a string naming a scipy.spatial distance. can be in
        ['euclidian', 'minkowski', 'seuclidiean', 'sqeuclidean', 'chebyshev'
         'cityblock', 'cosine', 'correlation', 'hamming', 'jaccard',
         'Bray-Curtis', 'mahalanobis', 'yule', 'matching', 'dice', 'kulsinski',
         'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath',
         'canberra', 'wminkowski'])
    p : double
        The p-norm to apply (for Minkowski, weighted and unweighted)
    Return
    ------
    k_star: int: optimal number of cluster
    """
    nb_data, nb_feature = X.shape
    # if no maximum number of clusters set, take datasize divided by 2
    if not k_max:
        k_max = nb_data // 2

    Y = - nb_feature / 2
    info_gain = 0
    old_dist = pow(
        distortion(X, np.zeros(nb_data), distortion_meth, p) / nb_feature, Y)
    for k in range(2, k_max + 1):
        cluster_estimator.set_params(n_clusters=k)
        labs = cluster_estimator.fit_predict(X)
        new_dist = pow(
            distortion(X, labs, distortion_meth, p) / nb_feature, Y)
        if new_dist - old_dist >= info_gain:
            k_star = k
            info_gain = new_dist - old_dist
        old_dist = new_dist
    return k_star
