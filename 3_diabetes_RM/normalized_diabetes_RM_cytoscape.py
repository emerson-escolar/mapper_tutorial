import numpy as np
import sklearn
import sklearn.decomposition as skd

import pandas

import kmapper as km
import mapperutils.linkage_gap as lk
import mapperutils.visualization as viz
import mapperutils.filters as filt

import mapperutils.exporter as expo


raw = pandas.read_csv("../0_data/diabetes_RM/data.txt",
                      delim_whitespace=True, index_col = 0,
                      usecols=lambda x: x not in {'Chap','Sec','number'})

target = raw.loc[:,'Clinical_classification']
data = raw.drop('Clinical_classification',axis = 1)
normalized_data = (data - data.mean())/data.std()
mapper = km.KeplerMapper(verbose=0)


def do_analysis(data, lens, name_prefix, nc, po,metric='euclidean'):
    graph = mapper.map(lens,
                       data.values,
                       clusterer = lk.LinkageGap(verbose=0,metric=metric),
                       cover=km.Cover(n_cubes=nc, perc_overlap=po))

    name = "{}_n{}_o{}".format(name_prefix,nc, po)
    mapper.visualize(graph,
                     # color_function=target.values,
                     color_function=lens,
                     path_html=name + "_nm_diabetes_RM.html",
                     title=name + "nm_diabetes_RM")


    nx_graph_simple = expo.kmapper_to_nxmapper(graph)
    expo.cytoscapejson_dump(nx_graph_simple, name + "_nm_diabetes_RM_simple.cyjs")


    extra_data = {x: list(raw.loc[:,x]) for x in raw.columns}
    extra_normalized_data = {"normalized_" + x: list(normalized_data.loc[:,x])
                             for x in normalized_data.columns}
    extra_data.update(extra_normalized_data)

    extra_transforms = {x : np.mean for x in extra_data
                        if x != "Clinical_classification"}

    nx_graph = expo.kmapper_to_nxmapper(graph, node_extra_data = extra_data,
                                        node_transforms = extra_transforms)
    expo.cytoscapejson_dump(nx_graph, name + "_nm_diabetes_RM.cyjs")


p = 5
eps = 4
for nc in range(7,8):
    dens = filt.gauss_kernel_density(normalized_data.values, epsilon=eps)
    do_analysis(normalized_data, dens, "dens"+"{:.1f}".format(eps), nc, p*0.1)
