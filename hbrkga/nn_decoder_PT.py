from hbrkga.brkga_mp_ipr.types import BaseChromosome
from hbrkga.nn_instance_PT import NNInstance

class NNDecoder():

    def __init__(self, instance: NNInstance, limits = [(1000,2000), (2000,4000), (2000,6000), (0.000001,0.1), (0,0.001)]):
        
        self._instance = instance
        self._limits = limits

    ###########################################################################

    def decode(self, chromosome: BaseChromosome, rewrite: bool, percentage: float = 1.0) -> float:

        hyper_parameters = [0,0,0,0,0]
        for geneIdx in range(len(chromosome)):
            hyper_parameters[geneIdx] = (chromosome[geneIdx] * (self._limits[geneIdx][1]-self._limits[geneIdx][0])) + self._limits[geneIdx][0]
        return self._instance.train(
            layer1 = round(hyper_parameters[0]), #Number of neurons in the first hidden layer
            layer2 = round(hyper_parameters[1]), #Number of neurons in the second hidden layer
            layer3 = round(hyper_parameters[2]), #Number of neurons in the third hidden layer
            learning_rate = hyper_parameters[3], #learning rate
            rr = hyper_parameters[4], #beta parameter to regularization. 0 to ignore
            percentage=percentage) #percentage of the training data to be used; e.g. 1.0 means all of the data
