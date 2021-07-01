import numpy as np
import sklearn
import sklearn.decomposition as skd

import pandas

import kmapper as km
import mapperutils.linkage_gap as lk
import mapperutils.visualization as viz
import mapperutils.filters as filt

raw = pandas.read_csv("../0_data/diabetes_RM/data.txt",
                      delim_whitespace=True, index_col = 0,
                      usecols=lambda x: x not in {'Chap','Sec','number'})

target = raw.loc[:,'Clinical_classification']
data = raw.drop('Clinical_classification',axis = 1)

print(data)
print("********** Standard deviation **********")
print(data.std())

print("********** Min **********")
print(data.min())

print("********** Max **********")
print(data.max())


normalized_data = (data - data.mean())/data.std()
print()
print("########## NORMALIZED DATA ##########")
print("********** Standard deviation **********")
print(normalized_data.std())

print("********** Min **********")
print(normalized_data.min())

print("********** Max **********")
print(normalized_data.max())


mapper = km.KeplerMapper(verbose=0)
def do_analysis(data, lens, name_prefix, nc, po,metric='euclidean'):
    graph = mapper.map(lens,
                       data.values,
                       clusterer = lk.LinkageGap(verbose=0,metric=metric),
                       cover=km.Cover(n_cubes=nc, perc_overlap=po))

    name = "{}_n{}_o{}".format(name_prefix,nc, po)
    mapper.visualize(graph,
                     # color_values=target.values,
                     color_values=lens,
                     color_function_name=name_prefix,
                     path_html=name + "_nm_diabetes_RM.html",
                     title=name + "nm_diabetes_RM")



nc = 5
po = 5
eps = 0.5

dens = filt.gauss_kernel_density(normalized_data.values, epsilon=eps)
do_analysis(normalized_data, dens, "dens"+"{:.1f}".format(eps), nc, po*0.1)
