import numpy as np
import sklearn
import pandas

import sklearn.decomposition as skd

import kmapper as km
import mapperutils.linkage_gap as lk
import mapperutils.visualization as viz
import mapperutils.filters as filt

import mapperutils.exporter as expo

mapper = km.KeplerMapper(verbose=0)

def do_analysis(data, lens, cf, name_prefix, nc, po):
    name = "{:s}_n{:d}_o{:.2f}".format(name_prefix, nc, po)
    graph = mapper.map(lens,
                       data.values,
                       clusterer = lk.LinkageGap(verbose=0, metric='euclidean'),
                       cover=km.Cover(n_cubes=nc, perc_overlap=po))
    mapper.visualize(graph,
                     color_function = cf,
                     path_html = name + "_travel.html",
                     title= name+"_travel")

    extra_data = {x: list(data.loc[:,x]) for x in data.columns}
    extra_transforms = {x : np.mean for x in extra_data}

    nx_graph = expo.kmapper_to_nxmapper(graph, node_extra_data = extra_data,
                                        node_transforms = extra_transforms)
    expo.cytoscapejson_dump(nx_graph, name + "_travel.cyjs")


raw = pandas.read_csv("../0_data/travel/tripadvisor_review.csv", index_col = 0)
category_map = {
    "Category 1": "art_galleries",
    "Category 2": "dance_clubs",
    "Category 3": "juice_bars",
    "Category 4": "restaurants",
    "Category 5": "museums",
    "Category 6": "resorts",
    "Category 7": "parks/picnic_spots",
    "Category 8": "beaches",
    "Category 9": "theaters",
    "Category 10": "religious_institutions"}
raw = raw.rename(columns = category_map)
normalized = (raw - raw.mean()) / raw.std()
data = normalized

# Filter function:
lens = skd.PCA(n_components=2).fit_transform(data.values)

# FOR USE IN COLOR in output:
cf = np.array(data.mean(axis=1))


for n in range(4,11,2):
    for p in range(35,51,5):
        do_analysis(data, lens, cf, "euc_pca2d" + "_mean", n, 0.01*p)
