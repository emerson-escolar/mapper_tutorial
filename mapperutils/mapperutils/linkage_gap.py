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
    if metric == 'precomputed':
        return -sklearn.metrics.silhouette_score(scipy.spatial.distance.squareform(X), labels, metric=metric)
    else:
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


    def fit(self, X, y=None):
        """Fit the Linkage clustering on data

        Parameters
        ----------
        X : ndarray
            A collection of `m` observation vectors in `n` dimensions
            as an `m` by `n` array.

            Alternatively, a condensed distance matrix. A condensed distance matrix
            is a flat array containing the upper triangular of the distance matrix.
            This is the form that ``pdist`` returns. All elements of the condensed
            distance matrix must be finite, i.e. no NaNs or infs.

        y : ignored

        Returns
        -------
        self
        """

        if len(X.shape) == 2 and X.shape[0] == 1:
            self.labels_ = np.array([1])
            return

        Z = scipy.cluster.hierarchy.linkage(X, method=self.method, metric=self.metric)

        # MAPPER PAPER GAP HEURISTIC
        gap_heuristic_percentiles = {'firstgap': 0, 'midgap': 50, 'lastgap':100}
        if self.heuristic in gap_heuristic_percentiles:
            percentile = gap_heuristic_percentiles[self.heuristic]
            self.labels_, k = mapper_gap_heuristic(Z, percentile, self.k_max)
        else:
            # do something
            pass

        # FINAL REPORTING
        if self.verbose:
            print("{} clusters detected in {} points".format(k,X.shape[0]))
            if k <= 1:
                print("silhouette score: invalid, too few final clusters")
            else:
                print("silhouette score: {}".format(-negative_silhouette(X, self.labels_, metric=self.metric)))
        return self
