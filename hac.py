
import sys
import random as rand
import math
import pdb
from utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

#============#
# HAC        #
#============#

def distance_wrapper(args):
    fn = args['fn']
    return fn(args['c1'], args['c2'], args['d'])

def HAC(data, K, metric):
    CMIN = 0
    CMAX = 1
    CMEAN = 2
    CCENT = 3
    fn = cmin
    if metric == CMAX:
        fn = cmax
    elif metric == CMEAN:
        fn = cmean
    elif metric == CCENT:
        fn = ccent

    m = len(data[0])

    clusters = map(lambda e: [e], data)
    while (len(clusters) != K):
        # find pairs of clusters
        cluster_pairs = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                assert(i != j)
                args = {'fn': fn, 'c1': clusters[i], 'c2': clusters[j], 'd': squareDistance}
                cluster_pairs.append(args)

        #pdb.set_trace()
        closest_pair = argmin(cluster_pairs, distance_wrapper)
        c1 = closest_pair['c1']
        c2 = closest_pair['c2']
        clusters.remove(c1)
        clusters.remove(c2)
        c1.extend(c2)
        clusters.append(c1)

    for c in clusters:
        print "Cluster length: " + str(len(c))
        totals = [0.]*m
        for example in c:
            for i in range(m):
                totals[i] += example[i]
        mean = map(lambda total: total / len(c), totals)
        print "Cluster mean: " + str(mean)

    return clusters

def scatterplot(clusters, K, N, metric):
    results = plt.figure(figsize=(10,7))
    axes = results.add_subplot(111, projection='3d')
    colors = ["seagreen", "deeppink", "deepskyblue", "darkorange"]
    plots = []
    rects = []
    labels = []
    for i in range(K):
        cluster = clusters[i]
        xs = [example[0] for example in cluster]
        ys = [example[1] for example in cluster]
        zs = [example[2] for example in cluster]
        label = "Cluster size: " + str(len(cluster))
        labels.append(label)
        plots.append(axes.scatter(xs, ys, zs, c=colors[i], label=label))
        rects.append(patches.Rectangle((0, 0), 1, 1, color=colors[i]))
    #legend((rects[0], rects[1], rects[2], rects[3]), (labels[0], labels[1], labels[2], labels[3]), loc="best")
    #handles, labels = axes.get_legend_handles_labels()
    axes.legend(rects, labels, loc=3, prop={'size':10})
    plot_name = "HAC Clustering for K=" + str(K) + ", N=" + str(N) + ", " + metric + " distance metric"
    plt.title(plot_name)

    file_name = "hac_k" + str(K) + "_n_" + str(N) + "_" + metric
    results.savefig(file_name + ".eps", dpi=600)
