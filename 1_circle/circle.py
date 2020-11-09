import math

import numpy as np
import sklearn
import kmapper as km

import mapperutils.visualization as viz

N = 1000
noiselevel = 0.05

# generate random points on circle of radius 1
angle = math.pi * 2 * np.random.rand(N)
data = np.stack((np.cos(angle), np.sin(angle)), axis=1) + np.random.normal((0,0), noiselevel, (N,2))

data3 = np.zeros((N,3))
data3[:,:-1] = data


# initialize mapper
mapper = km.KeplerMapper(verbose=2)

def do_analysis(lens, name_prefix):

    viz.scatter3d(data3, lens, show=False)
    viz.plt.savefig(name_prefix + "circle.png")
    viz.plt.close("all")

    graph = mapper.map(lens,
                       data,
                       clusterer=sklearn.cluster.DBSCAN(eps=0.1, min_samples=5),
                       cover=km.Cover(n_cubes=10, perc_overlap=0.2))
    mapper.visualize(graph, color_values=lens,
                     path_html = name_prefix + "_circle_output.html",
                     title = name_prefix + " circle",
                     lens = lens)

# try a bunch of filter functions f
lens = mapper.fit_transform(data, projection=[0],scaler=None)
print(data)
do_analysis(lens, "xproj")

lens = mapper.fit_transform(data, projection=[1],scaler=None)
do_analysis(lens, "yproj")

lens = mapper.fit_transform(data, projection="std",scaler=None)
do_analysis(lens, "std")


lens = np.angle( data[:,0] + 1.j*data[:,1] ).reshape(N,1)
do_analysis(lens, "angle")

# warning: if projection is not understood, kmapper quietly uses identity map as f
