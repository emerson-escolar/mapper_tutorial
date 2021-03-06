import pytest
import scipy.cluster.hierarchy
import scipy.spatial.distance as spd
import numpy as np

import mapperutils.linkage_gap as lk

# artificial data with "obvious" clustering
X = np.array([[0,0,0],
              [0,1,0],
              [1,0,0],
              [0,0,1],
              [100,0,0],
              [100,1,0],
              [101,0,0],
              [100,0,1],
              [0,0,100],
              [0,1,100],
              [1,0,100],
              [0,0,101]])


def test_num_clusters():
    X = np.random.normal(size=(400,2))

    Z = scipy.cluster.hierarchy.linkage(X, method='single', metric='euclidean')
    merge_distances = Z[:,2]

    for i,t in enumerate(reversed(merge_distances)):
        labels = scipy.cluster.hierarchy.fcluster(Z, t, criterion='distance')

        assert len(set(labels)) == i+1
        assert t == lk.cluster_number_to_threshold(len(set(labels)), merge_distances)


def test_heuristics():
    fg = lk.LinkageGap(heuristic='firstgap').fit(X)
    assert len(np.unique(fg.labels_)) == 3

    with pytest.raises(Exception):
        fg = lk.LinkageGap(heuristic='foobar')


def test_heuristics_precomputed():
    dists = spd.squareform(spd.pdist(X))

    fg = lk.LinkageGap(heuristic='firstgap', metric='precomputed').fit(dists)
    assert len(np.unique(fg.labels_)) == 3
