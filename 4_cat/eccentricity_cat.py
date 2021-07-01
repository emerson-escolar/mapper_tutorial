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
                       clusterer = lk.LinkageGap(verbose=0),
                       cover=km.Cover(n_cubes=nc, perc_overlap=po))
    mapper.visualize(graph,
                     color_values=lens,
                     color_function_name=name_prefix,
                     path_html = name + "_cat.html",
                     title= name + "_cat");

n = 12
p = 0.5

# ********** reference cat **********
cat = trimesh.load_mesh("../0_data/cat/cat-reference-simplified.obj")
data = cat.vertices
lens = filt.eccentricity(data,p=1)

viz.scatter3d(data, lens,colorsMap='viridis')
do_analysis(data, lens, "cat_ecc", n, p)

# ********** seated cat **********
cat = trimesh.load_mesh("../0_data/cat/cat-02-simplified.obj")
data = cat.vertices

lens = filt.eccentricity(data,p=1)
viz.scatter3d(data, lens,colorsMap='viridis')
do_analysis(data, lens, "seated_cat_ecc", n, p)
