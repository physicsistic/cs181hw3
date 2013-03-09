# clust_impl.py
# ------------
# Implementation of clustering algorithms for hw3.

import sys
import random

class Example():
    def __init__(self, location):
        self.location = location
        self.prototype = None
        self.m = len(location)

    # squared Euclidian norm
    def d_squared(self, prototype):
        d = 0
        assert(len(self.location) == len(prototype))
        for i in range(len(self.location)):
            d += (self.location[i] - prototype[i])**2
        return d

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
                old_prototype.examples.remove(example)
            closest_protype.examples.append(example)
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
        totals = [0.]*self.m
        for example in self.examples:
            for i in range(self.m):
                totals[i] += example[i]
        mean = map(lambda total: total / m, totals)
        self.location = mean

    def to_string(self):
        string = "<"
        for component in self.location:
            string += str(component) + " "
        string += ">"
        return string


def k_means(data, K):
    # sample data to determine initial prototype assignment
    prototypes = map(lambda p: Prototype(p), random.sample(data, K))
    examples = map(lambda e: Example(e), data)
    reassignments = -1
    iterations = 0
    while (reassignments != 0):
        iterations += 1
        count = 0
        # assign examples to closest prototypes
        for example in examples:
            if (example.adjust_prototype(prototypes)):
                count += 1
        # adjust prototype locations to mean of examples
        for prototype in prototypes:
            prototype.adjust_location()

        reassignments = count

    print "Iterations: " + str(iterations)
    print "Cluster means:"
    for prototype in prototypes:
        print prototype.to_string()

    # compute mean squared error
    total_squared_error = 0.
    for prototype in prototypes:
        for example in prototype.examples:
            total_squared_error += example.d_squared(prototype)
    mean_squared_error = total_squared_error / len(examples)

    return mean_squared_error
                
