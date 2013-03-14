
import sys
import random as rand
import math
from copy import deepcopy
import pdb
from utils import *

#===============#
# AutoClass     #
#===============#


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

    # divide attributes into continuous and discrete
    continuous = [0, 9, 10, 44, 45, 46]
    discrete = [i for i in range(48) if i not in continuous]
    m_c = len(continuous)
    m_d = len(discrete)

    # initialize arrays
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
            for k in range(K):
                gamma[n][k] = theta_pi[k] * total_prob(examples[n], theta_bern[k], theta_mu[k], \
                        theta_sigma[k], m, continuous) / math.fsum([theta_pi[j] \
                        * total_prob(examples[n], theta_bern[j], theta_mu[j], \
                        theta_sigma[j], m, continuous) for j in range(K)])
        
        N_hat = [math.fsum([gamma[n][k] for n in range(N)]) for k in range(K)]

        # M-step
        old_theta_pi = deepcopy(theta_pi)
        old_theta_bern = deepcopy(theta_bern)
        old_theta_mu = deepcopy(theta_mu)
        old_theta_sigma = deepcopy(theta_sigma)
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
