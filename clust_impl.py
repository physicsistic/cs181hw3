# clust_impl.py
# ------------
# Implementation of clustering algorithms for hw3.

import sys
import random
import pdb
from utils import *

class Example():
    def __init__(self, location):
        self.location = location
        self.prototype = None
        self.m = len(location)

    # squared Euclidian norm
    def d_squared(self, prototype):
        return squareDistance(self.location, prototype.location)

    # Assigns example to closest prototype
    # Returns True if the example's prototype was changed
    def adjust_prototype(self, prototypes):
        reassignment = False
        old_prototype = self.prototype
        # find closest prototype
        closest_distance = sys.maxint
        closest_prototype = prototypes[0] 
        for prototype in prototypes:
            d = self.d_squared(prototype)
            if d < closest_distance:
                closest_distance = d
                closest_prototype = prototype
        # switch prototype assignment if necessary
        if (old_prototype != closest_prototype):
            if old_prototype:
                old_prototype.examples.remove(self)
            closest_prototype.examples.append(self)
            self.prototype = closest_prototype
            reassignment = True
        return reassignment


class Prototype():
    def __init__(self, location):
        self.location = location
        self.examples = []
        self.m = len(location)

    # move prototype to mean of examples
    def adjust_location(self):
        if (len(self.examples) > 0):
            totals = [0.]*self.m
            for example in self.examples:
                for i in range(self.m):
                    totals[i] += example.location[i]
            mean = map(lambda total: total / len(self.examples), totals)
            self.location = mean

    def to_string(self):
        string = "<"
        for component in self.location:
            string += str(component) + " "
        string += ">"
        return string

def mean_squared_error(prototypes):
    total_squared_error = 0.
    N = 0
    for prototype in prototypes:
        for example in prototype.examples:
            N += 1
            total_squared_error += example.d_squared(prototype)
    return total_squared_error / N

def k_means(data, K):
    # sample data to determine initial prototype assignment
    prototypes = map(lambda p: Prototype(p), random.sample(data, K))
    examples = map(lambda e: Example(e), data)
    reassignments = -1
    iterations = 0
    while (reassignments != 0):
        iterations += 1
        print iterations
        count = 0
        # assign examples to closest prototypes
        for example in examples:
            if (example.adjust_prototype(prototypes)):
                count += 1
        # adjust prototype locations to mean of examples
        for prototype in prototypes:
            prototype.adjust_location()

        reassignments = count
        print "Reassignments: " + str(count)
        print "MSE: " + str(mean_squared_error(prototypes))

    print "Iterations: " + str(iterations)
    print "Cluster info:"
    for prototype in prototypes:
        #print prototype.to_string()
        print "Size: " + str(len(prototype.examples))

    return mean_squared_error(prototypes)

#========#
# HAC    #
#========#

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
        
                
