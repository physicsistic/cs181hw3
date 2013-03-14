# clust.py
# -------
# Lexi Ross and Ye Zhao
#
# Note: implementations of clustering algorithms can be found
# in clust_impl.py.

import sys
import random as rand
import k_means
import hac
import autoclass

DATAFILE = "adults-small.txt"

#validateInput()

def validateInput():
    if len(sys.argv) != 3:
        return False
    if sys.argv[1] <= 0:
        return False
    if sys.argv[2] <= 0:
        return False
    return True


#-----------


def parseInput(datafile):
    """
    params datafile: a file object, as obtained from function `open`
    returns: a list of lists

    example (typical use):
    fin = open('myfile.txt')
    data = parseInput(fin)
    fin.close()
    """
    data = []
    for line in datafile:
        instance = line.split(",")
        instance = instance[:-1]
        data.append(map(lambda x:float(x),instance))
    return data


def printOutput(data, numExamples):
    for instance in data[:numExamples]:
        print ','.join([str(x) for x in instance])

# main
# ----
# The main program loop
# You should modify this function to run your experiments

def main():
    # Validate the inputs
    if(validateInput() == False):
        print "Usage: clust numClusters numExamples"
        sys.exit(1);

    numClusters = int(sys.argv[1])
    numExamples = int(sys.argv[2])

    #Initialize the random seed
    
    rand.seed()

    #Initialize the data

    
    dataset = file(DATAFILE, "r")
    if dataset == None:
        print "Unable to open data file"


    data = parseInput(dataset)
    
    
    dataset.close()
    #printOutput(data,numExamples)

    # ==================== #
    # WRITE YOUR CODE HERE #
    # ==================== #

    #sample = rand.sample(data, numExamples)
    #result = k-means.k_means(sample, numClusters)
    #print "Mean squared error: " + str(result)

    sample = rand.sample(data, numExamples)
    min_clusters = hac.HAC(sample, numClusters, 0)
    max_clusters = hac.HAC(sample, numClusters, 1)
    hac.scatterplot(min_clusters, numClusters, numExamples, "min")
    hac.scatterplot(max_clusters, numClusters, numExamples, "max")

    #sample = rand.sample(data, numExamples)
    #result = autoclass.auto_class(sample, numClusters, 0.00001)

if __name__ == "__main__":
    validateInput()
    main()
