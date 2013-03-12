# clust_impl.py
# ------------
# Implementation of clustering algorithms for hw3.

import sys
import random as rand
import math
from numpy import *
import pdb
from utils import *

#================#
# K-means        #
#================#

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
    prototypes = map(lambda p: Prototype(p), rand.sample(data, K))
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
        
#===============#
# AutoClass     #
#===============#

#class Parameter():
    #def __init__(self, m):
        #self.c = 0.5
        #self.d1 = [0]*m
        #self.d0 = [0]*m

    ## distance metric for determining convergence
    #def compare(other):
        #vector1 = [self.c] + self.d1 + self.d0
        #vector2 = [other.c] + other.d1 + other.d0
        #return squareDistance(vector1, vector2)

class Parameter():
    def __init__(self, K, m):
        self.pi = 1.0/K
        self.mu = matrix([[0.]*m]) # means
        self.sigma = matrix([[0.]*m]) # covariances

#def covariance(example, mean):
    #return (example - mean) * (example - mean).T

def sample_covariance(examples, mean, m):
    N = len(mean)
    #variances = [sum((example[i] - mean[i])*(example[i] - mean[i]) for example in examples)/N for i in range(m)]
    variances = [0.]*m
    for i in range(m):
        for n in range(N):
            variances[i] += (examples[n][0, i] - mean[0, i])**2
        variances[i] /= N
    return matrix([variances])

def normal(x, mean, var):
    return (1 / (var * math.sqrt(2 * math.pi))) * math.exp(-.5 * ((x - mean) / var)**2)

def multi_normal(example, param, m):
    density = 1
    for i in range(m):
        density *= normal(example[0, i], param.mu[0, i], param.sigma[0, i])
    return density

def auto_class(examples, K, threshold):
    m = len(examples[0])
    N = len(examples)

    examples = [matrix([e]) for e in examples]

    # set initial values
    old_theta = [Parameter(K, m)]*K
    theta = [Parameter(K, m)]*K

    for k in range(K):
        # pick a random data point for cluster mean
        r = rand.randint(0, N - 1)
        theta[k].mu = examples[r]
        # set cluster variance to data covariance matrix
        theta[k].sigma = sample_covariance(examples, theta[k].mu, m)
    gamma = [[0.]*K]*N


    converged = False
    iterations = 0
    while(not converged): 
        iterations += 1
        # E-step
        for n in range(N):
            for k in range(K):
                pdb.set_trace()
                gamma[n][k] = theta[k].pi * multi_normal(examples[n], theta[k], m) / \
                        math.fsum([theta[j].pi * multi_normal(examples[n], theta[j], m) for j in range(K)])
        
        N_hat = [math.fsum([gamma[n][k] for n in range(N)]) for k in range(K)]

        # M-step
        old_theta = theta
        for k in range(K):
            theta[k].pi = N_hat[k] / N
            theta[k].mu = (1 / N_hat[k]) * sum([gamma[n][k] * examples[n] for n in range(N)])
            theta[k].sigma = (1 / N_hat[k]) * sum([gamma[n][k] * (examples[n] - theta[k].mu) \
                    * (examples[n] - theta[k].mu).T for n in range(N)])

        # normalize mixing coefficients
        total = sum([theta[k].pi for k in range(K)])
        for k in range(K):
            theta[k].pi /= total

        # check convergence
        theta_vector = [theta[k].pi for k in range(K)] + sum([theta[k].mu.tolist() for k in range(K)]) \
                + sum([theta[k].sigma.tolist() for k in range(K)])
        old_theta_vector = [old_theta[k].pi for k in range(K)] + sum([old_theta[k].mu.tolist() for k in range(K)]) \
                + sum([old_theta[k].sigma.tolist() for k in range(K)])
        distance = math.sqrt(sum((x - y)**2 for (x,y) in zip(theta_vector, old_theta_vector)))
        print "Iterations: " + str(iterations) + " Distance: " + str(distance)
        if (distance < threshold):
            converged = True

    return theta




                
