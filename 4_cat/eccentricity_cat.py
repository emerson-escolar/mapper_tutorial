import numpy as np
import sklearn

import trimesh

import kmapper as km
import mapperutils.linkage_gap as lk
import mapperutils.visualization as viz
import mapperutils.filters as filt


mapper = km.KeplerMapper(verbose=0)


def do_analysis(data, lens, name_prefix, nc, po):
    name = "{}_n{}_o{}".format(name_prefix, nc, po)
    graph = mapper.map(lens,
                       data,
                       clusterer = # ****** (1) ******,
                       cover=km.Cover(n_cubes=nc, perc_overlap=po))
    mapper.visualize(graph,
                     color_function=lens,
                     path_html = name + "_cat.html",
                     title= name + "_cat")

n = 12
p = 0.5

# ********** reference cat **********
cat = trimesh.load_mesh("../0_data/cat/cat-reference-simplified.obj")
data = cat.vertices
lens = filt.eccentricity(data,p=1)

viz.scatter3d(data, lens,colorsMap='viridis')
do_analysis(data, lens, "cat_ecc_", n, p)

# ********** seated cat **********
cat = trimesh.load_mesh("../0_data/cat/cat-02-simplified.obj")
data = # ****** (2) ******

lens = # ****** (3) ******
viz.scatter3d(data, lens,colorsMap='viridis')
do_analysis(data, lens, "seated_cat_ecc_", n, p)


