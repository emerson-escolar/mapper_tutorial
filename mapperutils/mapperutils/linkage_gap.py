import sklearn.cluster
import sklearn.base
import sklearn.metrics

import numpy as np

import scipy.cluster.hierarchy
import scipy.spatial.distance


def find_histogram_gap(merge_distances, percentile, bins='doane'):
    hist, bin_edges = np.histogram(merge_distances, bins=bins)

    if np.alltrue(hist != 0):
        return None

    gaps = np.argwhere(hist == 0).flatten()
    idx = np.percentile(gaps, percentile, interpolation='nearest')
    threshold = bin_edges[idx]
    return threshold


def mapper_gap_heuristic(Z, percentile, k_max=None):
    merge_distances = Z[:,2]

    if k_max != None and k_max != np.inf:
        merge_distances = merge_distances[-k_max:]

    threshold = find_histogram_gap(merge_distances,percentile,'doane')
    if threshold == None:
        labels = np.ones(Z.shape[0]+1)
        k = 1
    else:
        labels = scipy.cluster.hierarchy.fcluster(Z, t=threshold, criterion='distance')
        k = len(set(labels))

    return labels, k



def negative_silhouette(X, labels, metric):
    """
    Compute the negative of the silhouette score.
    Uses sklearn.metrics.negative_silhouette

    Parameters
    ----------
    X : array [n_samples, n_samples] if metric == "precomputed", or
        array [n_samples, n_features] otherwise

        Array of pairwise distances between samples, or a feature array.

    labels : array, shape = [n_samples]
         Predicted labels for each sample.

    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`metrics.pairwise.pairwise_distances
        <sklearn.metrics.pairwise.pairwise_distances>`. If X is the distance
        array itself, use ``metric="precomputed"``.
    """

    return -sklearn.metrics.silhouette_score(X, labels, metric=metric)


class LinkageGap(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin):
    """
    Agglomerative Linkage Clustering with Mapper gap heuristic

    Uses scipy.cluster.hierarchy algorithms.
    Class is designed to emulate sklearn.cluster format, for compatibility with kmapper.

    Instead of having to specify n_clusters, uses a heuristic to determine number of clusters.

    Parameters
    ----------
    method : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}
        Which linkage method to use, as available in scipy.cluster.hierarchy.linkage.

    metric : str or function, optional
        See the ``scipy.spatial.distance.pdist`` function for a list of valid distance metrics.
        A custom distance function can also be used.

    heuristic : {"firstgap", "midgap", "lastgap"}
        Which heuristic to use to determine number of clusters.
        first/mid/last gap is based on the original Mapper paper.

    k_max : int, optional
        Maximum number of clusters.

    Attributes
    ----------
    labels_ : array [n_samples]
        cluster labels for each point

    Notes
    -----
    This is essentially a wrapper around scipy.cluster.hierarchy.linkage.
    In particular, warnings about that algorithm apply here too.

    Quoted:
    "Methods 'centroid', 'median' and 'ward' are correctly defined only if
    Euclidean pairwise metric is used. If `X` is passed as precomputed
    pairwise distances, then it is a user responsibility to assure that
    these distances are in fact Euclidean, otherwise the produced result
    will be incorrect."
    """

    def __init__(self, method='single', metric='euclidean', heuristic='firstgap',
                 k_max=None, verbose=1):

        self.method = method
        self.metric = metric
        self.heuristic = heuristic
        self.verbose = verbose

        if k_max == None:
            k_max = np.inf
        self.k_max = k_max

        if self.heuristic not in ['firstgap', 'midgap', 'lastgap']:
            raise RuntimeError("Invalid heuristic {}!".format(self.heuristic))


    def fit(self, X, y=None):
        """Fit the Linkage clustering on data

        Parameters
        ----------
        X : array [n_samples, n_samples] if metric == "precomputed", or
            array [n_samples, n_features] otherwise

            Array of pairwise distances between samples, or a feature array.

        y : ignored

        Returns
        -------
        self
        """

        if len(X.shape) == 2 and X.shape[0] == 1:
            self.labels_ = np.array([1])
            return

        if self.metric != 'precomputed':
            Z = scipy.cluster.hierarchy.linkage(X, method=self.method, metric=self.metric)
        else:
            compdists = scipy.spatial.distance.squareform(X, force='tovector')
            Z = scipy.cluster.hierarchy.linkage(compdists, method=self.method, metric=self.metric)

        # MAPPER PAPER GAP HEURISTIC
        gap_heuristic_percentiles = {'firstgap': 0, 'midgap': 50, 'lastgap':100}
        percentile = gap_heuristic_percentiles[self.heuristic]
        self.labels_, k = mapper_gap_heuristic(Z, percentile, self.k_max)

        # FINAL REPORTING
        if self.verbose:
            print("{} clusters detected in {} points".format(k,X.shape[0]))
            if k <= 1:
                print("silhouette score: invalid, too few final clusters")
            else:
                print("silhouette score: {}".format(-negative_silhouette(X, self.labels_, metric=self.metric)))
        return self



def cluster_number_to_threshold(k, merge_distances):
    # check merge distances is non decreasing:
    assert np.all(np.diff(merge_distances) >= 0)

    # threshold is kth entry counting from the last.
    return (merge_distances[-k] if k <= len(merge_distances) else -np.inf)
