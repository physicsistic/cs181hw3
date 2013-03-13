# clust_impl.py
# ------------
# Implementation of clustering algorithms for hw3.

import sys
import random as rand
import math
from copy import deepcopy
import numpy as grumpy
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

#class Parameter():
    #def __init__(self, K, m):
        #self.pi = 1.0/K
        #self.mu = grumpy.matrix([[0.]*m]) # means
        #self.sigma = grumpy.matrix([[0.]*m]) # covariances

#def covariance(example, mean):
    #return (example - mean) * (example - mean).T

def sample_covariance(examples, mean, m, range_c):
    N = len(examples)
    variances = [0.]*m
    for i in range_c:
        variances[i] = math.fsum([(examples[n][i] - mean[i])**2 for n in range(N)])/N
    return variances

def gaussian(x, mean, var):
    if var == 0:
        return 1
    else:
        norm = (1 / math.sqrt(2 * math.pi * var)) * math.exp(-1*(x - mean)**2 / 2*var)
        #pdb.set_trace()
        #if norm > 1:
            #print "X: " + str(x)
            #print "Mean: " + str(mean)
            #print "Var: " + str(var)
            #print norm
        return norm

def bernoulli(x, bern):
    if x == 0:
        return 1 - bern
    else:
        return bern

def total_prob(example, bern, mu, sigma, m, continuous):
    density = 1
    for i in range(m):
        if i in continuous:
            density *= gaussian(example[i], mu[i], sigma[i])
        else:
            density *= bernoulli(example[i], bern[i])
    return density

def list_sum(lists):
    m = len(lists[0])
    total = [0.]*m
    for i in range(m):
        for l in lists:
            total[i] += l[i]
    return total

def inner_product_square(example, mean):
    return [(x - y)**2 for x, y in zip(example, mean)]

def auto_class(examples, K, threshold):
    m = len(examples[0])
    N = len(examples)
    rand.seed()

    # set initial values
    #old_theta = [Parameter(K, m) for k in range(K)]
    #theta = [Parameter(K, m) for k in range(K)]

    # divide attributes into continuous and discrete
    continuous = [0, 9, 10, 44, 45, 46]
    discrete = [i for i in range(48) if i not in continuous]
    m_c = len(continuous)
    m_d = len(discrete)

    theta_pi = [0.]*K
    theta_bern = [[0.]*m for k in range(K)]
    theta_mu = [[0.]*m for k in range(K)]
    theta_sigma = [[0.]*m for k in range(K)]

    for k in range(K):
        theta_pi[k] = 1./K
        # initialize Bernoulli parameters based on expectation of attributes
        for i in discrete:
            if i <= 8:
                theta_bern[k][i] = 1./8
            elif i <= 17:
                theta_bern[k][i] = 1./7
            elif i <= 31:
                theta_bern[k][i] = 1./14
            elif i <= 37:
                theta_bern[k][i] = 1./6
            elif i <= 42:
                theta_bern[k][i] = 1./5
            else:
                theta_bern[k][i] = 0.5
        # pick a random data point for cluster mean
        r = rand.randint(0, N - 1)
        print r
        #pdb.set_trace()
        for i in continuous:
            theta_mu[k][i] = examples[r][i]
        # set cluster variance to data covariance matrix
        theta_sigma[k] = sample_covariance(examples, theta_mu[k], m, continuous)

    # initialize probabilities
    gamma = [[0.]*K for n in range(N)]


    converged = False
    iterations = 0
    while(not converged): 
        iterations += 1
        # E-step
        for n in range(N):
            #print "Gamma before: " + str(gamma[n])
            for k in range(K):
                #print "Before: " + str(gamma[n][k])
                gamma[n][k] = theta_pi[k] * total_prob(examples[n], theta_bern[k], theta_mu[k], \
                        theta_sigma[k], m, continuous) / math.fsum([theta_pi[j] \
                        * total_prob(examples[n], theta_bern[j], theta_mu[j], \
                        theta_sigma[j], m, continuous) for j in range(K)])
                #print "After: " + str(gamma[n][k])
            #print "Gamma after: " + str(gamma[n])
        
        N_hat = [math.fsum([gamma[n][k] for n in range(N)]) for k in range(K)]

        # M-step
        old_theta_pi = deepcopy(theta_pi)
        old_theta_bern = deepcopy(theta_bern)
        old_theta_mu = deepcopy(theta_mu)
        old_theta_sigma = deepcopy(theta_sigma)
        # gamma[n][k] * examples[n] ==> [gamma[n][k] * examples[n][i] for i in d]
        for k in range(K):
            theta_pi[k] = N_hat[k] / N
            # update bernoulli params
            for i in discrete:
                theta_bern[k][i] = 0
                for n in range(N):
                    theta_bern[k][i] += gamma[n][k] * examples[n][i]
                theta_bern[k][i] *= (1 / N_hat[k])
            # update gaussian means
            for i in continuous:
                theta_mu[k][i] = 0
                for n in range(N):
                    theta_mu[k][i] += gamma[n][k] * examples[n][i]
                theta_mu[k][i] *= (1 / N_hat[k])
            # update gaussian variances
            for i in continuous:
                theta_sigma[k][i] = 0
                for n in range(N):
                    theta_sigma[k][i] += gamma[n][k] * inner_product_square(examples[n], theta_mu[k])[i]
                theta_sigma[k][i] *= (1 / N_hat[k])
                
        # normalize mixing coefficients
        #total = sum([theta[k].pi for k in range(K)])
        #for k in range(K):
            #theta[k].pi /= total

        # check convergence
        theta_vector = []
        old_theta_vector = []
        for k in range(K):
            theta_vector.append(theta_pi[k])
            old_theta_vector.append(old_theta_pi[k])
        for k in range(K):
            theta_vector += theta_mu[k]
            old_theta_vector += old_theta_mu[k]
        for k in range(K):
            theta_vector += theta_sigma[k]
            old_theta_vector += old_theta_sigma[k]

        distance = math.sqrt(sum([(x - y)**2 for (x,y) in zip(theta_vector, old_theta_vector)]))
        print "Iterations: " + str(iterations) + " Distance: " + str(distance)
        if (distance < threshold):
            converged = True

    return theta_pi




                
