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

def axisEqual3D(ax):
    # https://stackoverflow.com/a/19248731
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

viz.scatter3d(data, lens, colorsMap='viridis', show=False);
viz.plt.gca().view_init(elev=90,azim=0)
axisEqual3D(viz.plt.gca())
viz.plt.show()



for n in range(5,15):
    for p in range(2,6):
        graph = mapper.map(lens, data,
                           clusterer=lk.LinkageGap(verbose=0),
                           #clusterer=sklearn.cluster.DBSCAN(eps=0.5,min_samples=1),
                           cover=km.Cover(n_cubes=n, perc_overlap=p*0.1));
        name = "n{}_p{}".format(n,p)
        mapper.visualize(graph, color_function=lens,
                         path_html="hand_" + name + ".html",title="hand, " +name);
