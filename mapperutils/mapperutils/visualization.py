import matplotlib.colors
import matplotlib.pyplot as plt

import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

"""
Helpful(?) scripts to do plotting
"""

def scatter3d(X, lens, colorsMap='jet_r', show=True, colorbar=True, fig=None):
    # Derived from https://stackoverflow.com/a/27179208
    assert X.shape[1] == 3
    assert lens.shape[1] == 1
    assert lens.shape[0] == X.shape[0]

    cs = lens[:,0].tolist()
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    color_function = scalarMap.to_rgba(cs)

    if fig is None:
        fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:,0], X[:,1], X[:,2], c=color_function)

    scalarMap.set_array(cs)

    if colorbar:
        fig.colorbar(scalarMap)

    if show:
        plt.show()

    return color_function


def axisEqual3D(ax):
    # https://stackoverflow.com/a/19248731
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def qs_scatter(x,y, *args, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x,y, *args, **kwargs)

def qs_scatter3(xs,ys,zs, *args, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection="3d")
    return ax.scatter(xs,ys,zs, *args, **kwargs)


def qs_plot(matrix, **kwargs):
    if len(matrix.shape) != 2:
        return False
    if matrix.shape[1] == 2:
        return qs_scatter(matrix[:,0],matrix[:,1], **kwargs)
    elif matrix.shape[1] == 3:
        return qs_scatter3(matrix[:,0],matrix[:,1], matrix[:,2], **kwargs)
    return False
