import trimesh
import networkx as nx
import numpy as np


def compute_distances(mesh):
    edges = mesh.edges_unique
    length = mesh.edges_unique_length

    G = nx.Graph()
    for edge, L in zip(edges, length):
        G.add_edge(*edge, length=L)

    N = len(G)
    dist_mat = np.full( (N,N), np.inf)

    dist = nx.all_pairs_dijkstra_path_length(G)
    for idx, data in dist:
        for jdx, w in data.items():
            dist_mat[idx][jdx] = w

    return dist_mat


cat = trimesh.load_mesh("../0_data/cat/cat-reference-simplified.obj")
dist_mat = compute_distances(cat)
np.save("cat_dists", dist_mat)

seated_cat = trimesh.load_mesh("../0_data/cat/cat-02-simplified.obj")
seated_dist_mat = compute_distances(seated_cat)
np.save("seated_cat_dists", seated_dist_mat)
