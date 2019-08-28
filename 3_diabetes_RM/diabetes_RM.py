import numpy as np
import sklearn
import sklearn.decomposition as skd

import pandas

import kmapper as km
import mapperutils.linkage_gap as lk
import mapperutils.visualization as viz

import mappertools.filters as filt

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

quit()

mapper = km.KeplerMapper(verbose=0)
def do_analysis(data, lens, name_prefix, nc, po,metric='euclidean'):
    graph = mapper.map(lens,
                       data.values,
                       clusterer = lk.LinkageGap(verbose=0,metric=metric),
                       cover=km.Cover(n_cubes=nc, perc_overlap=po))

    name = "{}_n{}_o{}".format(name_prefix,nc, po)
    mapper.visualize(graph,
                     color_function=target.values,
                     path_html=name + "_diabetes_RM.html",
                     title=name + "diabetes_RM")



for nc in range(3,10):
    for po in range(5,6):
        for eps in range(1,4):
            dens = filt.gauss_kernel_density(normalized_data.values, epsilon=0.1*eps)
            do_analysis(normalized_data, dens, "dens"+str(eps), nc, po*0.1)
