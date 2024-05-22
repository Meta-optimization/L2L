import time
from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee
from network import NestBenchmarkNetwork

HPCBMOptimizeeParameters = namedtuple(
    'HPCBMOptimizeeParameters', [])


class HPCBMOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation


    def create_individual(self):
        """
        Creates and returns the individual
        """
        # create individual
        #synaptic_delay =
        #weight_excitatory =
        #weight_inhibitory = 
        #connectivita = 
        
        individual = {}
        return individual
    

    def bounding_func(individual):
        """
        """
        return individual
    


    def simulate(self, traj):
        """
        """
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation

        # TODO: think about which parameters should be accesible from top-level
        #   experiment specific:
        #       scale
        #       dt
        #       ...
        #   given by optimizer: 
        #       weight_excitatory
        #       weight_inhibitory
        #       ...


        scale = 0.1
        NE = int(9000 * scale)
        NI = int(2250 * scale)
        # number of incoming excitatory connections
        CE = int(1. * NE / scale)
        # number of incomining inhibitory connections
        CI = int(1. * NI / scale)
        delay = 1.5

        net = NestBenchmarkNetwork(NE, NI, CE, CI, weight_excitatory=1, weight_inhibitory=1, delay=delay)
        average_rate = net.run_simulation()

        # TODO: calculate fitness from average firing rate
        
        fitness = 0
        return (fitness,) 
    

    
    
    




