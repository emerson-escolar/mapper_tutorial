import trimesh
import numpy as np
import sklearn

import kmapper as km
import kmapper.jupyter as kmj

import mapperutils.linkage_gap as lk
import mapperutils.visualization as viz

mapper = km.KeplerMapper(verbose=2)

hand = trimesh.load_mesh("../0_data/hand/hand_simplified3k5.stl")
data = np.array(hand.vertices)
lens = data[:,1:2]

plot = True
if plot:
    viz.scatter3d(data, lens, colorsMap='viridis', show=False)
    viz.plt.gca().view_init(elev=90,azim=0)
    viz.axisEqual3D(viz.plt.gca())
    viz.plt.show()


n = 7
p = 0.2
graph = mapper.map(lens, data,
                   clusterer=lk.LinkageGap(verbose=0),
                   cover=km.Cover(n_cubes = n, perc_overlap = p))

name = "n{}_p{}".format(n,p)
mapper.visualize(graph, color_function=lens,
                 path_html="hand_only_" + name + ".html",title="hand, " +name);
