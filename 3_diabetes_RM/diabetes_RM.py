import numpy as np
import sklearn
import sklearn.decomposition as skd

import pandas

import kmapper as km
import mapperutils.linkage_gap as lk
import mapperutils.visualization as viz



raw = pandas.read_csv("../0_data/diabetes_RM/data.txt",
                      delim_whitespace=True, index_col = 0,
                      usecols=lambda x: x not in {'Chap','Sec','number'})

target = raw.loc[:,'Clinical_classification']
data = raw.drop('Clinical_classification',axis = 1)


mapper = km.KeplerMapper(verbose=0)
def do_analysis(data, lens, name_prefix, nc, po):
    graph = mapper.map(lens,
                       data.values,
                       clusterer = lk.LinkageGap(verbose=0),
                       cover=km.Cover(n_cubes=nc, perc_overlap=po))

    name = "{}_n{}_o{}".format(name_prefix,nc, po)
    mapper.visualize(graph,
                     color_function=target.values,
                     path_html=name + "_diabetes_RM.html",
                     title=name + "diabetes_RM")


lens = skd.PCA(n_components=2).fit_transform(data.values)
for nc in range(5,9):
    for po in range(3,7):
        do_analysis(data, lens, "pca", nc, po*0.1)


normalized_data = (data - data.mean())/data.std()
lens = skd.PCA(n_components=2).fit_transform(normalized_data.values)
for nc in range(5,9):
    for po in range(3,7):
        do_analysis(normalized_data, lens, "ndata_pca", nc, po*0.1)






# UNUSED:
# import mappertools.text_dump as tdump
# output_fname = name + "_diabetes_RM.cyjs"
# extra_data = {}
# extra_transforms = {}

# nxgraph = tdump.kmapper_to_nxmapper(graph,
#                                     extra_data, extra_data,
#                                     extra_transforms, extra_transforms)
# tdump.cytoscapejson_dump(nxgraph,str(output_fname))
