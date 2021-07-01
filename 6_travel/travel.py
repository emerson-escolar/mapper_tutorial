import numpy as np
import sklearn
import pandas

import sklearn.decomposition as skd

import kmapper as km
import mapperutils.linkage_gap as lk
import mapperutils.visualization as viz
import mapperutils.filters as filt


mapper = km.KeplerMapper(verbose=0)

def do_analysis(data, lens, cf, cf_name, name_prefix, nc, po):
    name = "{:s}_n{:d}_o{:.2f}".format(name_prefix, nc, po)
    graph = mapper.map(lens,
                       data,
                       clusterer = lk.LinkageGap(verbose=0, metric='euclidean'),
                       cover=km.Cover(n_cubes=nc, perc_overlap=po))
    mapper.visualize(graph,
                     color_values = cf,
                     color_function_name = cf_name,
                     path_html = name + "_travel.html",
                     title= name+"_travel")

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
# normalized = (raw - raw.mean()) / raw.std()
# data = normalized.values
data = raw

# Filter function:
lens = skd.PCA(n_components=2).fit_transform(data.values)

# FOR USE IN COLOR in output:
# cf = np.array(data.mean(axis=1))
cf = np.array(data.loc[:,"art_galleries"])


p = 5
n = 5
do_analysis(data.values, lens, cf, "art_galleries", "euc_pca2d" + "_art", n, 0.1*p)
