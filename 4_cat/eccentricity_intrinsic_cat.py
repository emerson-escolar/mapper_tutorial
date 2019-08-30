import numpy as np
import sklearn

import trimesh

import kmapper as km
import mapperutils.linkage_gap as lk
import mapperutils.visualization as viz
import mapperutils.filters as filt


mapper = km.KeplerMapper(verbose=0)

def do_analysis(data, dists, lens, name_prefix, nc, po):
    name = "{}_n{}_o{}".format(name_prefix, nc, po)
    graph = mapper.map(lens,
                       dists,
                       clusterer = lk.LinkageGap(verbose=0, metric="precomputed"),
                       precomputed=True,
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

dist_mat = np.load("cat_dists.npy")
lens = filt.eccentricity_from_dist(dist_mat,p=1)

viz.scatter3d(data, lens, colorsMap='viridis')
do_analysis(data, dist_mat, lens, "cat_ecc_ins_", n, p)

# ********** seated cat **********
cat = trimesh.load_mesh("../0_data/cat/cat-02-simplified.obj")
data = cat.vertices

seated_dist_mat = np.load("seated_cat_dists.npy")
lens = filt.eccentricity_from_dist(seated_dist_mat,p=1)

viz.scatter3d(data, lens, colorsMap='viridis')
do_analysis(data, seated_dist_mat, lens, "seated_cat_ecc_ins_", n, p)
