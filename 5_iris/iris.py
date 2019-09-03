import numpy as np
import sklearn

import pandas

import kmapper as km
import mapperutils.linkage_gap as lk
import mapperutils.visualization as viz
import mapperutils.filters as filt

import sklearn.decomposition as skd

import sklearn.datasets
raw = sklearn.datasets.load_iris()
# print(raw)
# print(raw.data.shape)

target = raw.target
data = raw.data

mapper = km.KeplerMapper(verbose=0)

def do_analysis(lens, name_prefix, nc, po):
    graph = mapper.map(lens,
                       data,
                       clusterer = lk.LinkageGap(verbose=0),
                       # clusterer=sklearn.cluster.DBSCAN(1, min_samples=0),
                       cover=km.Cover(n_cubes=nc, perc_overlap=po))

    name = "{}_n{}_o{}".format(name_prefix,nc, po)
    mapper.visualize(graph,
                     color_function=target,
                     path_html=name + "_iris.html",
                     title=name + "_iris")

lens = skd.PCA(n_components=2).fit_transform(data)
nc = 5
po = 5
do_analysis(lens, "pca", nc, po*0.01)
