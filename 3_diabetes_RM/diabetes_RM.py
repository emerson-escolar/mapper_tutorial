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
                     path_html=name + "_diabetes_RM.html",
                     title=name + "diabetes_RM")


for nc in range(3,8):
    for po in range(5,6):
        for eps in range(5,205,20):
            dens = filt.gauss_kernel_density(data.values, epsilon=eps)
            do_analysis(data, dens, "dens"+str(eps), nc, po*0.1)
        for eps in range(1,11):
            dens = filt.gauss_kernel_density(data.values, epsilon=0.1*eps)
            do_analysis(data, dens, "dens"+"{0:.2f}".format(0.1*eps), nc, po*0.1)
